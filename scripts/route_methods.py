#!/usr/bin/env python3
"""Method-level records of the public route manifest.

`parity/capabilities/route_manifest.toml` keeps one `[[route]]` per query and graph class.
A licensed route groups many estimators, structures and validations, so it cannot say which
target, inference, uncertainty procedure and artifact consumer belongs to which method. This
script appends one `[[method]]` per method: a licensed (query, graph class, estimator,
inference) and each graphless, execution-only or fitting-gap route (already one method). Every
record is derived from the implementation and its evidence, then re-verified:

  estimator      its wire id is a closed `EstimatorId` name; the id's identifiers and provenance
                 pair come from `strategy_table/ids.rs`
  inference      the estimator's inference side is the one `compiler_consistency.rs` pins
                 (refused under the other mode)
  target         the query is on the support axes and the route exists in the manifest
  uncertainty    a measured coverage record's interval method; else an interval method a
                 named evidence test asserts; else a point-only cell; else the sibling
                 coordinates of the same estimator; else a declared procedure that the owning
                 source corroborates. The record names which, so nothing is a default.
  consumer       a non-test source that reads the artifact, with the test that exercises it
  evidence       tests that exist, are registered and name the method

    python3 scripts/route_methods.py             # rewrite the generated section
    python3 scripts/route_methods.py --check     # fail when stale or when a record fails verification
    python3 scripts/route_methods.py --report    # basis and kind counts
    python3 scripts/route_methods.py --self-test
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import provenance_links  # noqa: E402
from method_evidence import (  # noqa: E402
    INTERVAL_KIND,
    INTERVAL_REF,
    TOKEN,
    VARIANT_METHOD,
    Evidence,
)

MANIFEST = "parity/capabilities/route_manifest.toml"
SUPPORT_LICENSED = "parity/support_licensed.toml"
LICENSED_ROUTES = "parity/licensed_routes.toml"
SUPPORT_GRAPHLESS = "parity/support_graphless.toml"
SUPPORT_AXES = "parity/support_axes.toml"
CHILDREN = "parity/method_children/children.toml"
IDS = "crates/antecedent/src/strategy_table/ids.rs"
CONSISTENCY = "crates/antecedent/tests/compiler_consistency.rs"
MARK = (
    "# ---- generated method records (scripts/route_methods.py); do not edit below this line ----"
)
ANALYSIS_CONSUME = (
    "crates/antecedent-io-core/src/contract_section/consume.rs::consume_analysis_result"
)
ANALYSIS_DECODE = (
    "crates/antecedent-io-core/src/analysis_result_artifact/mod.rs::decode_analysis_result_artifact"
)
STRUCTURE_ORDER = ("explicit", "accepted", "graph_posterior")
KINDS = ("analytic", "design_aware_resampling", "posterior", "enclosure", "none")

# The closed vocabulary of graphless interval claims, each with the procedure that forms it.
GRAPHLESS_CLAIM_KIND = {
    "exposure_contrast_pointwise_95_student_interval": "analytic",
    "rmst_and_horizon_survival_pointwise_95_percentile_intervals": "design_aware_resampling",
    "simultaneous_95_survival_difference_band": "design_aware_resampling",
    "simultaneous_95_cumulative_incidence_difference_band": "design_aware_resampling",
    "policy_reference_incremental_pointwise_95_normal_intervals": "analytic",
    "pointwise_95_student_interval": "analytic",
    "pointwise_95_normal_interval": "analytic",
    "paired_policy_incremental_pointwise_95_normal_intervals": "analytic",
    "paired_policy_incremental_and_all_uplift_pointwise_95_normal_intervals": "analytic",
    "paired_policy_incremental_and_all_cate_pointwise_95_normal_intervals": "analytic",
    "finite_class_regret_simultaneous_95_normal_interval": "analytic",
    "post_adoption_event_time_pointwise_95_normal_intervals": "analytic",
    "three_pointwise_95_normal_intervals": "analytic",
    "all_action_pointwise_95_normal_intervals": "analytic",
    "conditional_q_pointwise_95_normal_interval": "analytic",
    "intercept_and_period_effects_pointwise_95_normal_intervals": "analytic",
}
# Declared procedures of the executing routes that no coverage record measures. Each is the
# procedure the owning source implements; `CORROBORATION` names what that source must contain.
DECLARED_KIND = {
    "bayesian.gcomp.model_based_arm_effect": "posterior",
    "gcm.model.intervention_programs": "none",
    "gcm.model.mechanism_selection": "none",
    "gcm.attribution.intrinsic_influence": "design_aware_resampling",
    "gcm.attribute.bayesian": "posterior",
    "gcm.attribute.temporal.frequentist": "design_aware_resampling",
    "gcm.attribute.temporal.bayesian": "posterior",
    "quasi.synthetic_control": "none",
    "quasi.staggered_heterogeneity_robust": "design_aware_resampling",
    "quasi.covariate_adjusted_two_period": "design_aware_resampling",
    "quasi.augmented_panel_supplied_nuisance": "design_aware_resampling",
    "quasi.synthetic_control_conformal": "design_aware_resampling",
    "quasi.multi_unit_synthetic_placebo": "design_aware_resampling",
    "bayesian.staggered_heterogeneity_robust": "posterior",
    "bayesian.synthetic_control_predictive_path": "posterior",
    "bayesian.temporal_local_ratio": "posterior",
    "quasi.local_ratio.design_layer": "analytic",
    "quasi.temporal_local_ratio": "design_aware_resampling",
    "bayesian.panel_did": "posterior",
    "bayesian.synthetic_control": "posterior",
    "bayesian.fuzzy_rd": "posterior",
    "bayesian.regression_kink": "posterior",
    "bayesian.temporal.white_innovation_linear": "posterior",
    "temporal.glm_link_contrast": "analytic",
    "bayesian.temporal.glm_link_contrast": "posterior",
    "temporal.lagged_instrument": "analytic",
    "bayesian.temporal.lagged_instrument": "posterior",
    "bayesian.binary_policy_value": "posterior",
    "bayesian.sequential_regime_value": "posterior",
    "sequential.crossfit_contrast.wald": "analytic",
    "bayesian.sequential_crossfit_contrast": "posterior",
    "matching.period_history_contrast": "analytic",
    "bayesian.period_history_outcome_model": "posterior",
    "iv_bounds.balke_pearl": "enclosure",
    "bayesian.iv_bounds.endpoint_bootstrap": "posterior",
    "association.descriptive.static": "analytic",
    "association.descriptive.temporal": "analytic",
    "bayesian.association.descriptive.static": "posterior",
    "bayesian.association.descriptive.temporal": "posterior",
    "bayesian.randomized_finite_population": "posterior",
    "bayesian.randomized_superpopulation": "posterior",
    "bayesian.switchback_effect": "posterior",
    "bayesian.piecewise_survival": "posterior",
    "bayesian.continuous_dose_policy": "posterior",
    "bayesian.randomized_quantile": "posterior",
    "attribution.mechanism_change_detection": "none",
    "attribution.unit_change": "none",
    "discovery.static": "none",
    "discovery.temporal": "none",
    "gcm.attribute": "design_aware_resampling",
    "gcm.simulate": "none",
    "measurement.static.frequentist": "analytic",
    "measurement.static.bayesian": "posterior",
    "measurement.temporal.frequentist": "analytic",
    "measurement.temporal.bayesian": "posterior",
    "measurement.transport.pooled_differential": "analytic",
    "composition.static.frequentist": "none",
    "composition.static.bayesian": "posterior",
    "composition.temporal.frequentist": "none",
    "composition.temporal.bayesian": "posterior",
    "graphless.dose_grid.randomized_dose.local_quadratic": "analytic",
    "bayesian.dose_grid": "posterior",
    "bayesian.complier_effect": "posterior",
    "interference.cluster_bayesian_bootstrap": "posterior",
}
for _design in (
    "two_arm",
    "multi_arm",
    "categorical",
    "factorial",
    "interference",
    "continuous_dose",
    "smoothed_dose",
    "dose_grid",
):
    DECLARED_KIND[f"temporal_design.excursion.{_design}"] = "analytic"
    DECLARED_KIND[f"bayesian.temporal_design.excursion.{_design}"] = "posterior"
DECLARED_KIND["temporal_design.excursion.quantile"] = "design_aware_resampling"
DECLARED_KIND["bayesian.temporal_design.excursion.quantile"] = "posterior"
# The lag-0 average effect on a temporal graph reads the pulse at offset 0: its frequentist interval
# is the circular-block bootstrap of the pulse and its Bayesian reading is the per-completion
# posterior, so the route's procedure is the block resampling.
for _class in ("TemporalDag", "TemporalCpdag", "TemporalPag"):
    DECLARED_KIND[f"closed.AverageEffect.{_class}"] = "design_aware_resampling"
DECLARED_KIND["temporal_design.regime_survival"] = "design_aware_resampling"
DECLARED_KIND["bayesian.temporal_design.regime_survival"] = "posterior"
# Public methods that have no route record: reached through a facade or a library entry. Keyed by
# the implementing module and quadrant; the value is the uncertainty procedure the module implements.
FACADE_KIND = {
    "crates/antecedent-estimate/src/propensity/penalized.rs|static.frequentist": "analytic",
    "crates/antecedent-estimate/src/penalized_propensity_posterior.rs|static.bayesian": "posterior",
    "crates/antecedent-estimate/src/temporal_penalized_score.rs|temporal.frequentist": "analytic",
    "crates/antecedent-estimate/src/temporal_penalized_score.rs|temporal.bayesian": "posterior",
    "crates/antecedent-estimate/src/quantile.rs|static.frequentist": "none",
    "crates/antecedent-estimate/src/smoothed_dose.rs|static.frequentist": "design_aware_resampling",
    "crates/antecedent-estimate/src/smoothed_dose_posterior.rs|static.bayesian": "posterior",
    "crates/antecedent-estimate/src/adaptive_inference.rs|static.frequentist": "analytic",
    "crates/antecedent-stats/src/adaptive_inference/participant_posterior.rs|static.bayesian": "posterior",
    "crates/antecedent-stats/src/adaptive_inference/procedure_repeated_measures.rs|temporal.frequentist": "analytic",
    "crates/antecedent-stats/src/adaptive_inference/repeated_measures_posterior.rs|temporal.bayesian": "posterior",
    "crates/antecedent-estimate/src/categorical_treatment.rs|static.frequentist": "analytic",
    "crates/antecedent-estimate/src/categorical_treatment_posterior.rs|static.bayesian": "posterior",
    "crates/antecedent-estimate/src/augmented_panel_did.rs|temporal.frequentist": "design_aware_resampling",
    "crates/antecedent-estimate/src/bayesian_staggered_event_study.rs|temporal.bayesian": "posterior",
    "crates/antecedent-validate/src/target_native/mod.rs|static.frequentist": "design_aware_resampling",
    "crates/antecedent-validate/src/target_native/bayesian.rs|static.bayesian": "posterior",
    "crates/antecedent-validate/src/target_native/engine.rs|temporal.frequentist": "design_aware_resampling",
    "crates/antecedent-validate/src/target_native/bayesian.rs|temporal.bayesian": "posterior",
    "crates/antecedent-design/src/decision.rs|static.frequentist": "none",
    "crates/antecedent-design/src/evsi.rs|static.bayesian": "posterior",
    "crates/antecedent-design/src/temporal_policy_transport.rs|temporal.frequentist": "none",
    "crates/antecedent-design/src/evsi.rs|temporal.bayesian": "posterior",
    "crates/antecedent-state/src/ols_refresh.rs|static.frequentist": "none",
    "crates/antecedent-state/src/conjugate_refresh.rs|static.bayesian": "posterior",
    "crates/antecedent/src/period_refresh.rs|temporal.frequentist": "none",
    "crates/antecedent/src/period_refresh.rs|temporal.bayesian": "posterior",
    "crates/antecedent/src/analysis/execute/compile.rs|static.frequentist": "none",
    "crates/antecedent/src/analysis/execute/compile.rs|static.bayesian": "none",
    "crates/antecedent/src/analysis/execute/compile.rs|temporal.frequentist": "none",
    "crates/antecedent/src/analysis/execute/compile.rs|temporal.bayesian": "none",
    "scripts/closure_audit.py|static.frequentist": "none",
    "scripts/closure_audit.py|static.bayesian": "none",
    "scripts/closure_audit.py|temporal.frequentist": "none",
    "scripts/closure_audit.py|temporal.bayesian": "none",
    "crates/antecedent-estimate/src/joint_panel.rs|static.frequentist": "analytic",
    "crates/antecedent-estimate/src/joint_panel.rs|static.bayesian": "posterior",
    "crates/antecedent-design/src/program_search/continuous.rs|static.frequentist": "none",
    "crates/antecedent-design/src/program_search/continuous.rs|static.bayesian": "none",
    "crates/antecedent-design/src/program_search/continuous.rs|temporal.frequentist": "none",
    "crates/antecedent-design/src/program_search/continuous.rs|temporal.bayesian": "none",
    "crates/antecedent/src/analysis/joint_panel.rs|temporal.frequentist": "design_aware_resampling",
    "crates/antecedent/src/analysis/joint_panel.rs|temporal.bayesian": "posterior",
    "crates/antecedent-estimate/src/joint_panel/inference.rs|temporal.frequentist": "design_aware_resampling",
    "crates/antecedent-estimate/src/joint_panel/inference.rs|temporal.bayesian": "posterior",
    "crates/antecedent-estimate/src/joint_panel/time_varying.rs|temporal.frequentist": "design_aware_resampling",
    "crates/antecedent-estimate/src/joint_panel/time_varying.rs|temporal.bayesian": "posterior",
    "crates/antecedent/src/analysis/joint_panel/experiment.rs|temporal.bayesian": "posterior",
    "crates/antecedent-estimate/src/joint_panel/family.rs|static.frequentist": "analytic",
    "crates/antecedent-estimate/src/joint_panel/family.rs|static.bayesian": "posterior",
    "crates/antecedent-estimate/src/joint_panel/family.rs|temporal.frequentist": "design_aware_resampling",
    "crates/antecedent-estimate/src/joint_panel/family.rs|temporal.bayesian": "posterior",
}
# What the owning sources of a declared procedure must show. `none` needs no corroboration.
CORROBORATION = {
    "posterior": ("posterior", "Posterior", "Dirichlet", "credible", "bayesian"),
    "design_aware_resampling": (
        "bootstrap",
        "resample",
        "permutation",
        "multiplier",
        "conformal",
        "placebo",
    ),
    "analytic": (
        "standard_error",
        "se_analytic",
        "sandwich",
        "delta",
        "variance",
        "interval",
        "Wald",
        "hc1",
        "newey",
        "Newey",
    ),
    "enclosure": ("bound", "identified_set", "IdentifiedSet", "partial"),
}
# Sibling coordinates of an estimator whose cells carry no record of their own: the Bayesian
# cells that publish the linear-mechanism point on each identified completion.
DECLARED_UNMEASURED = {
    ("temporal.linear.adjustment", "Bayesian"): (
        ["posterior", "enclosure"],
        "crates/antecedent/tests/temporal_class_point_queries.rs::temporal_class_cross_world_points_match_the_linear_mechanism",
    ),
}


T = "crates/antecedent/tests/"
# Executing routes whose evidence the literal search cannot find: the tests that run them.
EVIDENCE_OVERRIDE = {
    "gcm.model.mechanism_selection": [
        T + "structure_compiler_routes.rs::wrapped_mechanisms_name_themselves_in_the_program"
    ],
    "gcm.attribution.intrinsic_influence": [
        "crates/antecedent-attribution/tests/intrinsic_influence.rs::a_bootstrap_with_data_quantifies_mechanism_fit_separately",
        T
        + "attribute_mechanism_posterior.rs::an_intrinsic_influence_has_a_posterior_over_the_mechanisms",
    ],
    "gcm.attribute.temporal.frequentist": [
        T + "temporal_attribution_route_evidence.rs::temporal_attribution_frequentist_known_truth"
    ],
    "gcm.attribute.temporal.bayesian": [
        T + "temporal_attribution_route_evidence.rs::temporal_attribution_bayesian_known_truth"
    ],
    "quasi.covariate_adjusted_two_period": [
        T
        + "quasi_rollout_routes.rs::covariate_adjusted_two_period_did_recovers_the_effect_under_conditional_trends"
    ],
    "quasi.augmented_panel_supplied_nuisance": [
        T
        + "augmented_panel_did_route.rs::augmented_panel_did_retains_known_truth_and_point_only_semantics"
    ],
    "quasi.multi_unit_synthetic_placebo": [
        T
        + "quasi_rollout_routes.rs::multi_unit_synthetic_did_recovers_a_staggered_effect_with_placebo_intervals"
    ],
    "bayesian.synthetic_control_predictive_path": [
        T
        + "bayesian_design_routes.rs::synthetic_control_posterior_recovers_the_effect_and_records_convergence"
    ],
    "quasi.local_ratio.design_layer": [
        T
        + "local_polynomial_ratio_route.rs::retained_fuzzy_rd_recovers_truth_and_round_trips_calibrated_interval",
        T
        + "quasi_rollout_routes.rs::fuzzy_rd_with_a_selected_bandwidth_and_diagnostics_recovers_the_effect",
    ],
    "attribution.mechanism_change_detection": [
        T
        + "attribute_mechanism_posterior.rs::mechanism_change_detection_has_no_posterior_and_says_why"
    ],
    "attribution.unit_change": [
        T + "structure_compiler_routes.rs::mechanism_and_unit_change_run_on_the_retained_path"
    ],
    "gcm.attribute": [
        T + "structure_compiler_routes.rs::attribution_and_simulation_equal_their_engines"
    ],
    "gcm.simulate": [
        T + "structure_compiler_routes.rs::attribution_and_simulation_equal_their_engines",
        T
        + "attribute_mechanism_posterior.rs::every_simulation_target_carries_refit_uncertainty_in_both_modes",
    ],
    "measurement.transport.pooled_differential": [
        "python/tests/test_pooled_differential_transport.py::test_heterogeneity_widens_and_a_target_sample_narrows_the_interval"
    ],
    "composition.static.frequentist": [
        T
        + "composed_uncertainty_facade.rs::a_composed_interval_opens_only_with_a_measured_record_at_its_coordinate"
    ],
    "composition.static.bayesian": [
        T
        + "joint_law_propagation_facade.rs::an_aligned_nonlinear_functional_matches_the_joint_moment_truth"
    ],
    "composition.temporal.frequentist": [
        T + "horizon_composition.rs::coupled_horizons_compose_with_the_spread_of_the_realizations"
    ],
    "composition.temporal.bayesian": [
        T
        + "horizon_composition.rs::the_compiled_temporal_response_retains_joint_horizons_in_both_modes"
    ],
}
for _class in ("TemporalDag", "TemporalCpdag", "TemporalPag"):
    EVIDENCE_OVERRIDE[f"closed.AverageEffect.{_class}"] = [
        T
        + "lag_zero_average_effect.rs::the_lag_zero_average_effect_is_the_explicit_pulse_and_stays_unlicensed",
        T
        + "lag_zero_average_effect.rs::every_lag_zero_average_effect_publishes_an_artifact_a_consumer_reads",
    ]


# Executing routes whose artifact the evidence tests do not read back: the production reader of
# each artifact, and the test that drives it where one exists.
CONSUMER_OVERRIDE = {
    "bayesian.gcomp.model_based_arm_effect": [
        ANALYSIS_CONSUME,
        T + "identity_contract.rs::target_population_moves_target_and_program_not_identification",
    ],
    "gcm.model.intervention_programs": [
        "crates/antecedent-io-core/src/policy_program_artifact.rs",
        T + "policy_python_parity.rs::a_discrete_parent_policy_has_portable_rust_python_bits",
    ],
    "gcm.model.mechanism_selection": [
        "crates/antecedent-io-core/src/mechanism_wire.rs::mechanisms_from_wire",
        T + "structure_compiler_routes.rs::wrapped_mechanisms_name_themselves_in_the_program",
    ],
    "gcm.attribution.intrinsic_influence": [
        ANALYSIS_CONSUME,
        T
        + "checked_attribution_route_evidence.rs::all_frequentist_static_dag_attribution_plans_are_sealed_and_dependency_refused",
    ],
    "gcm.attribute": [
        ANALYSIS_CONSUME,
        T
        + "checked_attribution_route_evidence.rs::all_frequentist_static_dag_attribution_plans_are_sealed_and_dependency_refused",
    ],
    "attribution.mechanism_change_detection": [
        ANALYSIS_CONSUME,
        T
        + "checked_attribution_route_evidence.rs::all_frequentist_static_dag_attribution_plans_are_sealed_and_dependency_refused",
    ],
    "attribution.unit_change": [
        ANALYSIS_CONSUME,
        T
        + "checked_attribution_route_evidence.rs::all_frequentist_static_dag_attribution_plans_are_sealed_and_dependency_refused",
    ],
    "gcm.simulate": [
        "crates/antecedent-io-core/src/policy_program_artifact.rs",
        T + "policy_python_parity.rs::a_discrete_parent_policy_has_portable_rust_python_bits",
    ],
    "discovery.static": [
        "crates/antecedent/src/analysis/structure_route.rs::accept_discovered",
        T
        + "structure_compiler_routes.rs::discovered_provenance_reaches_the_downstream_contract_and_program",
    ],
    "discovery.temporal": [
        "crates/antecedent/src/analysis/structure_route.rs::accept_discovered",
        T
        + "structure_compiler_routes.rs::discovered_provenance_reaches_the_downstream_contract_and_program",
    ],
    "association.descriptive.static": [
        ANALYSIS_CONSUME,
        T
        + "association_route.rs::every_association_counterpart_publishes_an_artifact_a_consumer_reads_the_terms_from",
    ],
    "association.descriptive.temporal": [
        ANALYSIS_CONSUME,
        T
        + "association_route.rs::every_association_counterpart_publishes_an_artifact_a_consumer_reads_the_terms_from",
    ],
    "bayesian.association.descriptive.static": [
        ANALYSIS_CONSUME,
        T
        + "association_route.rs::every_association_counterpart_publishes_an_artifact_a_consumer_reads_the_terms_from",
    ],
    "bayesian.association.descriptive.temporal": [
        ANALYSIS_CONSUME,
        T
        + "association_route.rs::every_association_counterpart_publishes_an_artifact_a_consumer_reads_the_terms_from",
    ],
    "measurement.static.frequentist": [
        "crates/antecedent-io-core/src/measurement_model_wire.rs",
        "crates/antecedent-estimate/tests/measurement_models.rs::artifact_replays_independently_and_refuses_semantic_mutation",
    ],
    "measurement.static.bayesian": [
        "crates/antecedent-io-core/src/measurement_model_wire.rs",
        "crates/antecedent-estimate/tests/measurement_models.rs::artifact_replays_independently_and_refuses_semantic_mutation",
    ],
    "measurement.transport.pooled_differential": [
        "crates/antecedent/src/analysis/transported_measurement.rs"
    ],
    "composition.static.frequentist": [
        "crates/antecedent/src/analysis/composition.rs",
        T + "identity_contract.rs::composition_artifact_reload_in_separate_process",
    ],
    "composition.static.bayesian": [
        "crates/antecedent/src/analysis/composition.rs",
        T + "identity_contract.rs::composition_artifact_reload_in_separate_process",
    ],
    "composition.temporal.frequentist": [
        "crates/antecedent/src/analysis/composition.rs",
        T + "cross_surface_horizon.rs",
    ],
    "composition.temporal.bayesian": [
        "crates/antecedent/src/analysis/composition.rs",
        T + "cross_surface_horizon.rs",
    ],
    "graphless.dose_grid.randomized_dose.local_quadratic": [
        "crates/antecedent-io/src/dose_grid_functional_artifact.rs::consume_typed",
        "crates/antecedent-io/tests/dose_grid_artifact.rs::a_declared_curvature_bound_seals_bias_aware_intervals_and_the_band",
    ],
}


# Executing-unlicensed coordinates that name a method but no support cell: the manifest route
# records them in its notes and the test below runs them.
EXECUTING_UNLICENSED = (
    {
        "query": "InterventionResponse",
        "graph": "CoDetermined",
        "estimator": "response.bayesian",
        "inference": "Bayesian",
        "evidence": [
            "crates/antecedent/tests/unlicensed_refuter_states_and_mechanism_bootstrap.rs::codetermined_bayesian_joint_level_executes_unlicensed"
        ],
    },
)


def quote(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def listing(items: list[str]) -> str:
    return "[" + ", ".join(quote(i) for i in items) + "]"


# ---------------------------------------------------------------------------------- sources
def parse_estimators(ids_text: str) -> dict[str, dict]:
    """Wire name -> {variant, identifiers, provenance} from the exhaustive estimator_data match."""
    start = ids_text.index("const fn estimator_data")
    body = ids_text[start:]
    constants = {
        m.group(1): re.findall(r"IdentifierId::(\w+)", m.group(2))
        for m in re.finditer(r"const (\w+): &\[IdentifierId\] =\s*&\[(.*?)\];", ids_text, re.S)
    }
    identifier_names = {}
    ident_start = ids_text.index("const fn identifier_data")
    for m in re.finditer(
        r"IdentifierId::(\w+) => IdentifierData \{(.*?)\n        \},", ids_text[ident_start:], re.S
    ):
        name = re.search(r'name: "([^"]+)"', m.group(2))
        if name:
            identifier_names[m.group(1)] = name.group(1)
    out: dict[str, dict] = {}
    for m in re.finditer(r"EstimatorId::(\w+) => EstimatorData \{(.*?)\n        \},", body, re.S):
        block = m.group(2)
        name = re.search(r'name: "([^"]+)"', block).group(1)
        compat = re.search(r"compatible:\s*(\w+)", block).group(1)
        pair = re.search(r'provenance:\s*\(\s*"([^"]+)",\s*"([^"]+)",?\s*\)', block)
        same = re.search(r'provenance:\s*same_step\("([^"]+)"\)', block)
        provenance = (pair.group(1), pair.group(2)) if pair else (same.group(1), same.group(1))
        out[name] = {
            "variant": m.group(1),
            "identifiers": [identifier_names.get(v, v) for v in constants.get(compat, [])],
            "provenance": provenance,
        }
    return out


def inference_classes(text: str) -> dict[str, str]:
    """Estimator wire id -> Frequentist | Bayesian from the pinned refusals of compiler_consistency."""
    out: dict[str, str] = {}
    for const, side in (
        ("REFUSED_UNDER_BAYESIAN", "Frequentist"),
        ("REFUSED_UNDER_FREQUENTIST", "Bayesian"),
    ):
        m = re.search(rf"const {const}: &\[&str\] = &\[(.*?)\];", text, re.S)
        for name in re.findall(r'"([^"]+)"', m.group(1) if m else ""):
            out[name] = side
    return out


class Sources:
    def __init__(self, ev: Evidence) -> None:
        self.ev = ev
        load = lambda rel: tomllib.loads(ev.text(rel))  # noqa: E731
        self.manifest = load(MANIFEST)
        self.routes = {r["id"]: r for r in self.manifest["route"]}
        self.cells = load(SUPPORT_LICENSED)["cell"]
        self.lowered = load(LICENSED_ROUTES)["route"]
        self.graphless = {
            (lic["family"], lic["design"], lic["method"]): lic
            for lic in load(SUPPORT_GRAPHLESS)["license"]
        }
        self.queries = set(load(SUPPORT_AXES)["queries"])
        self.children = load(CHILDREN)["child"]
        self.estimators = parse_estimators(ev.text(IDS))
        self.sides = inference_classes(ev.text(CONSISTENCY))
        self.records = {r["id"]: r for r in ev.coverage_records()}
        self.provenance = provenance_links.stems(ev.root)
        self._tests: list[tuple[str, str]] | None = None

    # ---- test discovery: the tests whose body names a literal -----------------------------
    def test_corpus(self) -> list[tuple[str, str]]:
        if self._tests is None:
            files: list[tuple[str, str]] = []
            for rel in sorted(self.ev.repo.corpus()):
                if rel.endswith((".rs", ".py")) and (
                    "/tests/" in rel or rel.startswith("scripts/test_")
                ):
                    files.append((rel, self.ev.text(rel)))
            self._tests = files
        return self._tests

    def tests_matching(self, requirement: list[list[str]], limit: int = 3) -> list[str]:
        """Tests whose body names every alternative group (one literal of each group)."""
        found: list[tuple[int, str]] = []
        for rel, body in self.test_corpus():
            if not all(any(lit in body for lit in group) for group in requirement):
                continue
            for name, block in split_tests(rel, body):
                if all(any(lit in block for lit in group) for group in requirement):
                    score = int("consume_analysis_result" in block) * 2 + int(
                        "encode_contracted_result" in block
                    )
                    found.append((-score, f"{rel}::{name}"))
        found.sort()
        return [token for _, token in found[:limit]]

    def tests_naming(self, literals: list[str], limit: int = 3) -> list[str]:
        """`path::name` of tests whose body contains one of `literals` (a wire id or variant)."""
        found: list[tuple[int, str]] = []
        for rel, body in self.test_corpus():
            if not any(lit in body for lit in literals):
                continue
            for name, block in split_tests(rel, body):
                if any(lit in block for lit in literals):
                    score = int("consume_analysis_result" in block) * 2 + int(
                        "encode_contracted_result" in body
                    )
                    found.append((-score, f"{rel}::{name}"))
        found.sort()
        return [token for _, token in found[:limit]]


def split_tests(rel: str, body: str) -> list[tuple[str, str]]:
    """(name, text) of each test function in a Rust or Python test file."""
    out: list[tuple[str, str]] = []
    if rel.endswith(".py"):
        spans = [(m.start(), m.group(1)) for m in re.finditer(r"^\s*def (test_\w+)", body, re.M)]
    else:
        spans = [
            (m.start(), m.group(1))
            for m in re.finditer(
                r"#\[(?:tokio::)?test[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub )?(?:async )?fn (\w+)",
                body,
            )
        ]
    for index, (start, name) in enumerate(spans):
        end = spans[index + 1][0] if index + 1 < len(spans) else len(body)
        out.append((name, body[start:end]))
    return out


# ------------------------------------------------------------------------------ derivation
def design_of(graph_class: str) -> str:
    return "temporal" if graph_class.startswith("Temporal") else "static"


def record_ids(cell: dict) -> set[str]:
    ids = set(cell.get("calibration", []) or [])
    for figure in cell.get("coverage_figures", []) or []:
        ids |= set(figure.get("cites", []))
    for estimator in cell.get("calibration_estimators", []) or []:
        ids |= set(estimator.get("calibration", []))
    return ids


def consumer_for(
    src: Sources, tests: list[str], ladder: list[list[list[str]]] | None = None
) -> list[str]:
    """The production reader of the artifact and the evidence test that exercises it.

    The named evidence tests come first. When none of them reads the artifact, a test that
    names the same method and consumes an artifact stands in for it. `ladder` runs from the
    most specific requirement to the least; a requirement is a list of alternatives that must
    all be named (each alternative is a list of literals, any one of which suffices).
    """
    found = _consumer(src, tests)
    if found:
        return found
    for requirement in ladder or []:
        for token in src.tests_matching(requirement, limit=60):
            found = _consumer(src, [token])
            if found:
                return found
    return []


def _consumer(src: Sources, tests: list[str]) -> list[str]:
    for token in tests:
        if "consume_analysis_result" in src.ev.text(token.split("::")[0]):
            return [ANALYSIS_CONSUME, token]
    for token in tests:
        if "decode_analysis_result_artifact" in src.ev.text(token.split("::")[0]):
            return [ANALYSIS_DECODE, token]
    return []


def licensed_methods(src: Sources) -> list[dict]:
    ev = src.ev
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for cell in src.cells:
        for estimator in cell["estimators"]:
            groups[(cell["query"], cell["graph_class"], estimator, cell["inference"])].append(cell)
    lowered: dict[tuple[str, str, str, str], tuple[set[str], set[str]]] = defaultdict(
        lambda: (set(), set())
    )
    for row in src.lowered:
        query, graph, _structure, inference, _validation = row["coordinate"].split(":")
        key = (query, graph, row["estimator"], inference)
        lowered[key][0].add(row["identifier"])
        lowered[key][1].add(row["lowering"])

    # measured kinds per (cell structure) and per estimator-level sibling
    def measured(cells: list[dict]) -> set[str]:
        kinds = set()
        for cell in cells:
            for rid in record_ids(cell):
                record = src.records.get(rid)
                kind = ev.record_kind(record) if record else None
                if kind:
                    kinds.add(kind)
        return kinds

    sibling: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for (_query, graph, estimator, inference), cells in groups.items():
        for structure in STRUCTURE_ORDER:
            sibling[(estimator, inference, design_of(graph), structure)] |= measured(
                [c for c in cells if c["structure"] == structure]
            )

    methods = []
    for (query, graph, estimator, inference), cells in sorted(groups.items()):
        route_id = f"licensed.{query}.{graph}"
        route = src.routes.get(route_id)
        structures = sorted({c["structure"] for c in cells}, key=STRUCTURE_ORDER.index)
        by_structure: dict[str, tuple[str, str]] = {}
        for structure in structures:
            members = [c for c in cells if c["structure"] == structure]
            kinds = measured(members)
            sources_: list[str] = []
            basis = ""
            if kinds:
                basis = "measured_record"
                ids = sorted({rid for c in members for rid in record_ids(c) if rid in src.records})
                sources_ = [f"coverage_record:{ids[0]}"]
            elif {c.get("calibration_reason") for c in members} == {"no_interval_reported"}:
                kinds, basis, sources_ = {"none"}, "point_only_cell", ["cell:no_interval_reported"]
            else:
                asserted = set()
                for c in members:
                    for ck in c["checked_execution"]:
                        if ck["estimator"] == estimator:
                            asserted |= {
                                INTERVAL_KIND[VARIANT_METHOD[v]]
                                for v in INTERVAL_REF.findall(ev.text(ck["test"]))
                            }
                asserted -= {"none"}
                inherited = sibling[(estimator, inference, design_of(graph), structure)]
                if asserted:
                    kinds, basis = asserted, "asserted_in_test"
                    first = next(
                        ck["test"]
                        for c in members
                        for ck in c["checked_execution"]
                        if ck["estimator"] == estimator
                    )
                    sources_ = [f"test:{first}"]
                elif inherited:
                    kinds, basis = inherited, "estimator_records"
                    sources_ = [f"estimator:{estimator}:{inference}:{design_of(graph)}:{structure}"]
                elif (estimator, inference) in DECLARED_UNMEASURED:
                    kinds_list, test = DECLARED_UNMEASURED[(estimator, inference)]
                    kinds, basis, sources_ = (
                        set(kinds_list),
                        "declared_unmeasured",
                        [f"test:{test}"],
                    )
            by_structure[structure] = (
                ",".join(sorted(kinds, key=KINDS.index)),
                basis + ("|" + sources_[0] if sources_ else ""),
            )
        primary_structure = next(s for s in STRUCTURE_ORDER if s in by_structure)
        primary_kinds, primary_basis = by_structure[primary_structure]
        primary = primary_kinds.split(",")[0] if primary_kinds else "unresolved"
        evidence: list[str] = []
        for cell in cells:
            for ck in cell["checked_execution"]:
                if ck["estimator"] == estimator:
                    token = f"{ck['test']}::{ck['assertion']}"
                    if token not in evidence:
                        evidence.append(token)
        evidence = evidence[:4]
        identifiers, lowerings = lowered.get((query, graph, estimator, inference), (set(), set()))
        info = src.estimators.get(estimator)
        implementation = implementation_files(src, estimator, info["variant"] if info else "")
        methods.append(
            {
                "id": f"method.{query}.{graph}.{estimator}.{inference.lower()}",
                "kind": "licensed_estimator",
                "route": route_id,
                "child": "",
                "target": query,
                "graph_class": graph,
                "design": design_of(graph),
                "estimator": estimator,
                "identifier": sorted(identifiers),
                "lowering": sorted(lowerings),
                "inference": inference.lower(),
                "design_time_model": route["design_time_model"] if route else "unlisted",
                "status": "licensed",
                "structures": structures,
                "validations": sorted({c["validation"] for c in cells}),
                "coordinates": len(cells),
                "uncertainty": primary,
                "uncertainty_structure": primary_structure,
                "uncertainty_basis": primary_basis,
                "uncertainty_by_structure": [f"{s}={k}|{b}" for s, (k, b) in by_structure.items()],
                "decision_ports": route["decision_ports"] if route else "unlisted",
                "owning_module": route["owning_module"].split(" ")[0] if route else "unlisted",
                "implementation": implementation,
                "provenance": list(info["provenance"]) if info else [],
                "evidence": evidence,
                "artifact_consumer": consumer_for(
                    src,
                    evidence,
                    [[wire_literals(estimator, info), [graph]], [wire_literals(estimator, info)]],
                ),
            }
        )
    for extra in EXECUTING_UNLICENSED:
        query, graph, estimator, inference = (
            extra["query"],
            extra["graph"],
            extra["estimator"],
            extra["inference"],
        )
        route_id = f"licensed.{query}.{graph}"
        route = src.routes[route_id]
        info = src.estimators.get(estimator)
        inherited = sibling[(estimator, inference, design_of(graph), "explicit")]
        kinds = ",".join(sorted(inherited, key=KINDS.index))
        evidence = list(extra["evidence"])
        methods.append(
            {
                "id": f"method.{query}.{graph}.{estimator}.{inference.lower()}",
                "kind": "licensed_estimator",
                "route": route_id,
                "child": "",
                "target": query,
                "graph_class": graph,
                "design": design_of(graph),
                "estimator": estimator,
                "identifier": [],
                "lowering": [],
                "inference": inference.lower(),
                "design_time_model": route["design_time_model"],
                "status": "execution_only",
                "structures": ["explicit"],
                "validations": ["none"],
                "coordinates": 1,
                "uncertainty": kinds.split(",")[0] if kinds else "unresolved",
                "uncertainty_structure": "explicit",
                "uncertainty_basis": "estimator_records|"
                + f"estimator:{estimator}:{inference}:{design_of(graph)}:explicit",
                "uncertainty_by_structure": [f"explicit={kinds}|estimator_records"],
                "decision_ports": route["decision_ports"],
                "owning_module": route["owning_module"].split(" ")[0],
                "implementation": implementation_files(
                    src, estimator, info["variant"] if info else ""
                ),
                "provenance": list(info["provenance"]) if info else [],
                "evidence": evidence,
                "artifact_consumer": consumer_for(
                    src,
                    evidence,
                    [[wire_literals(estimator, info), [graph]], [wire_literals(estimator, info)]],
                ),
            }
        )
    return methods


def wire_literals(estimator: str, info: dict | None) -> list[str]:
    return [f'"{estimator}"'] + ([f"EstimatorId::{info['variant']}"] if info else [])


def implementation_files(src: Sources, estimator: str, variant: str) -> list[str]:
    """Non-test sources that dispatch or implement this estimator (its wire id or variant)."""
    ev = src.ev
    base = ev.root / "crates"
    hits: list[tuple[int, str]] = []
    for path in sorted(base.glob("*/src/**/*.rs")):
        rel = path.relative_to(ev.root).as_posix()
        if rel.endswith("strategy_table/ids.rs"):
            continue
        body = ev.text(rel)
        score = 0
        if f'"{estimator}"' in body:
            score += 2
        if variant and f"EstimatorId::{variant}" in body:
            score += 2
        if score:
            weight = 0 if "/analysis/execute/" in rel else 1 if "antecedent-estimate" in rel else 2
            hits.append((weight * 10 - score, rel))
    hits.sort()
    return [rel for _, rel in hits[:3]]


def route_methods(src: Sources) -> list[dict]:
    ev = src.ev
    by_route: dict[str, list[dict]] = defaultdict(list)
    for child in src.children:
        by_route[child["route"]].append(child)
    methods = []
    for route in src.manifest["route"]:
        if route["status"] not in ("graphless", "execution_only", "fitting_gap"):
            continue
        rid = route["id"]
        owner = route["owning_module"].split(" ")[0]
        inference = (
            "frequentist"
            if route["inference"].startswith("Frequentist")
            and "Bayesian" not in route["inference"].split(";")[0]
            else "bayesian"
            if route["inference"].startswith("Bayesian")
            else "both"
            if "Bayesian" in route["inference"] and "Frequentist" in route["inference"]
            else "unlisted"
        )
        evidence: list[str] = []
        if route.get("retained_route_test"):
            evidence.append(route["retained_route_test"])
        for child in by_route.get(rid, []):
            evidence += [
                t for t in ev.tokens(child["fixtures"]) if TOKEN.match(t) and t not in evidence
            ]
        estimator = (
            rid
            if rid in src.estimators
            else next(
                (
                    e
                    for e in src.estimators
                    if re.search(rf"\bestimator {re.escape(e)}\b", route["notes"])
                ),
                "",
            )
        )
        evidence += [t for t in EVIDENCE_OVERRIDE.get(rid, []) if t not in evidence]
        if not evidence:
            literals = [rid] + ([estimator] if estimator and estimator != rid else [])
            variant = src.estimators[estimator]["variant"] if estimator in src.estimators else ""
            literals = [f'"{x}"' for x in literals] + (
                [f"EstimatorId::{variant}"] if variant else []
            )
            evidence = src.tests_naming(literals)
        evidence = [t for t in evidence if ev.fixture_findings(t) == []][:4]
        uncertainty, basis = "unresolved", ""
        if rid.startswith("graphless.") and rid not in DECLARED_KIND:
            note = re.search(r"method=([A-Za-z0-9_]+)", route["notes"])
            match = [
                lic
                for (f, _d, m), lic in src.graphless.items()
                if note and m == note.group(1) and rid.startswith(f"graphless.{f}.")
            ]
            if match and match[0]["inference_claim"] in GRAPHLESS_CLAIM_KIND:
                uncertainty = GRAPHLESS_CLAIM_KIND[match[0]["inference_claim"]]
                basis = f"graphless_claim:{match[0]['inference_claim']}"
        elif rid in DECLARED_KIND:
            uncertainty, basis = DECLARED_KIND[rid], "declared_and_corroborated"
        info = src.estimators.get(estimator)
        override = CONSUMER_OVERRIDE.get(rid, [])
        methods.append(
            {
                "id": f"method.{rid}",
                "kind": "route_method",
                "route": rid,
                "child": "",
                "target": route["target"],
                "graph_class": route["graph_class"],
                "design": "temporal"
                if route["graph_class"].startswith("Temporal") or "temporal" in rid
                else "static",
                "estimator": estimator or "unlisted",
                "identifier": info["identifiers"] if info else [],
                "lowering": [],
                "inference": inference,
                "design_time_model": route["design_time_model"],
                "status": route["status"],
                "structures": [],
                "validations": [],
                "coordinates": 0,
                "uncertainty": uncertainty,
                "uncertainty_structure": "none",
                "uncertainty_basis": basis,
                "uncertainty_by_structure": [],
                "decision_ports": route["decision_ports"],
                "owning_module": owner,
                "implementation": [owner] + implementation_extra(rid),
                "provenance": list(info["provenance"]) if info else [],
                "evidence": evidence,
                "artifact_consumer": override
                or consumer_for(
                    src,
                    evidence,
                    [[[f'"{rid}"'] + ([f'"{estimator}"'] if estimator else [])]],
                ),
            }
        )
    return methods


def children_covered(methods: list[dict], child: dict) -> bool:
    """Whether a route or estimator method already implements this child."""
    inference = child["id"].rsplit(".", 1)[-1]
    wires = {m["estimator"] for m in methods if m["estimator"] != "unlisted"}
    tokens = {t for t in re.split(r"[^A-Za-z0-9_.]+", child["inference"]) if t in wires}
    for method in methods:
        if method["inference"] not in (inference, "both", "unlisted"):
            continue
        if method["route"] == child["route"] and child["route"] != "unlisted":
            return True
        if method["estimator"] in tokens:
            return True
    return False


def facade_methods(src: Sources, methods: list[dict]) -> list[dict]:
    ev = src.ev
    out = []
    for child in src.children:
        module = child["module"].split(";")[0].strip().split(" ")[0]
        if module in ("none", "unlisted") or child["gap"] not in (
            "none",
            "validation",
            "implementation",
        ):
            continue
        if children_covered(methods, child):
            continue
        parts = child["id"].split(".")
        stem = Path(module).parent.name if Path(module).stem == "mod" else Path(module).stem
        evidence = [t for t in ev.tokens(child["fixtures"]) if ev.fixture_findings(t) == []]
        consumers = ev.tokens(child["consumer"])
        kind = FACADE_KIND.get(f"{module}|{parts[-2]}.{parts[-1]}", "unresolved")
        out.append(
            {
                "id": f"method.{stem}.{parts[-2]}.{parts[-1]}",
                "kind": "facade_method",
                "route": "unlisted",
                "child": child["id"],
                "target": child["target"],
                "graph_class": "unlisted",
                "design": parts[-2],
                "estimator": "unlisted",
                "identifier": [],
                "lowering": [],
                "inference": parts[-1],
                "design_time_model": "unlisted",
                "status": "execution_only",
                "structures": [],
                "validations": [],
                "coordinates": 0,
                "uncertainty": kind,
                "uncertainty_structure": "none",
                "uncertainty_basis": "declared_and_corroborated",
                "uncertainty_by_structure": [],
                "decision_ports": "unlisted",
                "owning_module": module,
                "implementation": [module],
                "provenance": [],
                "evidence": evidence,
                "artifact_consumer": consumers,
            }
        )
    return out


FIELDS = (
    "id",
    "kind",
    "route",
    "child",
    "target",
    "graph_class",
    "design",
    "estimator",
    "identifier",
    "lowering",
    "inference",
    "design_time_model",
    "status",
    "structures",
    "validations",
    "coordinates",
    "uncertainty",
    "uncertainty_structure",
    "uncertainty_basis",
    "uncertainty_by_structure",
    "decision_ports",
    "owning_module",
    "implementation",
    "provenance",
    "evidence",
    "artifact_consumer",
)


def render(methods: list[dict]) -> str:
    out = [
        MARK,
        "# One record per method: a licensed (query, graph class, estimator, inference), or a",
        "# graphless, execution-only or fitting-gap route. Derived from the support records, the",
        "# estimator table, the coverage records and the evidence tests, then verified by",
        "# `python3 scripts/route_methods.py --check`.",
        "",
    ]
    for method in methods:
        out.append("[[method]]")
        for field in FIELDS:
            value = method[field]
            out.append(
                f"{field} = {listing(value) if isinstance(value, list) else quote(value)}"
                if not isinstance(value, int)
                else f"{field} = {value}"
            )
        out.append("")
    return "\n".join(out)


def build(src: Sources) -> list[dict]:
    methods = licensed_methods(src) + route_methods(src)
    return methods + facade_methods(src, methods)


# ------------------------------------------------------------------------------ verification
def verify(src: Sources, methods: list[dict]) -> list[str]:
    ev = src.ev
    out: list[str] = []
    names = set(src.estimators)
    seen: set[str] = set()
    for m in methods:
        mid = m["id"]
        if mid in seen:
            out.append(f"{mid}: duplicate method id")
        seen.add(mid)
        if m["kind"] == "facade_method":
            if m["child"] not in {c["id"] for c in src.children}:
                out.append(f"{mid}: child {m['child']} is not a method child")
        elif m["route"] not in src.routes:
            out.append(f"{mid}: route {m['route']} is not in the manifest")
        if m["kind"] == "licensed_estimator":
            if m["target"] not in src.queries:
                out.append(f"{mid}: target {m['target']} is not on the support axes")
            if m["estimator"] not in names:
                out.append(f"{mid}: estimator {m['estimator']} is not a closed EstimatorId name")
            side = src.sides.get(m["estimator"])
            if side and side.lower() != m["inference"]:
                out.append(
                    f"{mid}: inference {m['inference']} contradicts compiler_consistency, which pins {m['estimator']} as {side}"
                )
            for impl in m["implementation"]:
                body = ev.text(impl)
                info = src.estimators.get(m["estimator"], {})
                if (
                    f'"{m["estimator"]}"' not in body
                    and f"EstimatorId::{info.get('variant', '')}" not in body
                ):
                    out.append(f"{mid}: implementation {impl} does not name {m['estimator']}")
            if not m["implementation"]:
                out.append(f"{mid}: no source dispatches or implements {m['estimator']}")
            family = (m["provenance"] + [""])[0]
            if not provenance_links.resolve(family, src.provenance):
                out.append(f"{mid}: provenance family {family!r} resolves to no record")
        else:
            if not ev.repo.exists(m["owning_module"]):
                out.append(f"{mid}: owning module {m['owning_module']} does not exist")
            if m["owning_module"] not in m["implementation"]:
                out.append(f"{mid}: implementation omits the owning module")
        if not m["evidence"]:
            out.append(f"{mid}: no evidence test is named")
        for token in m["evidence"]:
            out.extend(f"{mid}: {p}" for p in ev.fixture_findings(token))
        consumer = m["artifact_consumer"]
        if not consumer:
            out.append(f"{mid}: no artifact consumer is found in the evidence tests")
        for token in consumer:
            out.extend(f"{mid}: {p}" for p in ev.consumer_findings(token))
        if consumer and not any(not ev.is_test_path(ev.split(t)[0]) for t in consumer):
            out.append(f"{mid}: every consumer is a test")
        out.extend(verify_uncertainty(src, m))
    return out


def verify_uncertainty(src: Sources, m: dict) -> list[str]:
    ev = src.ev
    mid = m["id"]
    kind, basis = m["uncertainty"], m["uncertainty_basis"]
    if kind not in KINDS:
        return [
            f"{mid}: uncertainty is not derived ({kind!r}); no coverage record, asserted interval, point-only cell, sibling record or declared procedure supports it"
        ]
    label, _, detail = basis.partition("|")
    if m["kind"] == "licensed_estimator":
        if label not in (
            "measured_record",
            "point_only_cell",
            "asserted_in_test",
            "estimator_records",
            "declared_unmeasured",
        ):
            return [f"{mid}: uncertainty basis {label!r} is outside the vocabulary"]
        if label == "measured_record":
            rid = detail.removeprefix("coverage_record:")
            record = src.records.get(rid)
            if record is None or ev.record_kind(record) not in m["uncertainty_by_structure"][
                0
            ].split("=")[1].split("|")[0].split(","):
                return [f"{mid}: coverage record {rid} does not show {kind}"]
        if label == "asserted_in_test":
            path = detail.removeprefix("test:")
            kinds = {INTERVAL_KIND[VARIANT_METHOD[v]] for v in INTERVAL_REF.findall(ev.text(path))}
            if kind not in kinds:
                return [f"{mid}: {path} asserts no {kind} interval"]
        if label == "declared_unmeasured":
            for problem in ev.fixture_findings(detail.removeprefix("test:")):
                return [f"{mid}: {problem}"]
        return []
    if label.startswith("graphless_claim:"):
        claim = label.removeprefix("graphless_claim:")
        if GRAPHLESS_CLAIM_KIND.get(claim) != kind:
            return [f"{mid}: graphless claim {claim} is not a {kind} procedure"]
        return []
    if label != "declared_and_corroborated":
        return [f"{mid}: uncertainty basis {label!r} is outside the vocabulary"]
    declared = (
        FACADE_KIND.get(f"{m['owning_module']}|{m['design']}.{m['inference']}")
        if m["kind"] == "facade_method"
        else DECLARED_KIND.get(m["route"])
    )
    if declared != kind:
        return [f"{mid}: no declared procedure for {m['child'] or m['route']}"]
    if kind == "none":
        return []
    bodies = "".join(ev.text(f) for f in sources_of(ev, m))
    if not any(token in bodies for token in CORROBORATION[kind]):
        return [f"{mid}: the owning source shows no {kind} construction"]
    return []


# The estimator source behind an executor whose uncertainty it forms.
IMPLEMENTATION_EXTRA = {
    "association.descriptive.static": "crates/antecedent-estimate/src/association.rs",
    "association.descriptive.temporal": "crates/antecedent-estimate/src/association.rs",
    "bayesian.association.descriptive.static": "crates/antecedent-estimate/src/association.rs",
    "bayesian.association.descriptive.temporal": "crates/antecedent-estimate/src/association.rs",
}


def implementation_extra(rid: str) -> list[str]:
    extra = IMPLEMENTATION_EXTRA.get(rid, [])
    return [extra] if isinstance(extra, str) else list(extra)


# The graph-class routes that execute without a coverage record. Each runs the finite-discrete or
# regression estimator of its query on the identified functional of every completion, atom or
# window; the frequentist interval is the design-aware resample (row, cluster or circular block)
# of that functional and the Bayesian form is the Bayesian bootstrap over the same resamples, so
# the declared procedure is the design-aware resample the executing sources construct.
X = "crates/antecedent/src/analysis/execute/"
CLOSED_IMPLEMENTATION = {
    "AnomalyAttribution": [
        X + "attribution_path.rs",
        X + "class_attribution.rs",
        X + "temporal_attribution.rs",
    ],
    "ChangeAttribution": [
        X + "attribution_path.rs",
        X + "class_attribution.rs",
        X + "temporal_attribution.rs",
    ],
    "Counterfactual": [X + "class_counterfactual.rs", X + "cross_world_posterior.rs"],
    "MediationEffect": [X + "mediation_mixed.rs", X + "mediation_refutation.rs"],
    "TemporalMediationEffect": [
        X + "temporal_pag_mediation.rs",
        X + "temporal_admg_mediation.rs",
        X + "mediation_mixed.rs",
    ],
    "NestedCounterfactualEffect": [X + "class_path_specific.rs", X + "static_path.rs"],
    "PathSpecificEffect": [X + "class_path_specific.rs", X + "static_path.rs"],
    "InterventionalDistribution": [X + "class_distribution.rs"],
    "PulseEffect": [X + "temporal_admg_path.rs"],
    "SustainedEffect": [X + "temporal_admg_path.rs"],
    "TemporalConditionalPulse": [X + "temporal_admg_path.rs"],
    "TemporalConditionalSustained": [X + "temporal_admg_path.rs"],
    "InterferenceQuery": [X + "checked_interference.rs", X + "transport_interference_path.rs"],
    "ConditionalEffect": [X + "temporal_admg_path.rs", X + "atom_mix.rs"],
    "TransportQuery": [X + "transport_class.rs", X + "checked_transport_trial.rs"],
}
for _q in (
    "AverageDerivative",
    "PointDerivative",
    "DirectionalDerivative",
    "Elasticity",
    "SemiElasticity",
    "ResponseJacobian",
):
    CLOSED_IMPLEMENTATION[_q] = [X + "response_path.rs", X + "temporal_response_execution.rs"]
ARTIFACTS = T + "class_route_artifacts.rs::"
DERIVATIVE_STATIC = [
    T
    + "class_derivative_response.rs::frequentist_class_and_latent_confounded_derivatives_match_the_structural_reference",
    T
    + "class_derivative_response.rs::bayesian_class_and_latent_confounded_derivatives_publish_credible_intervals",
    T
    + "derivative_validation_on_classes.rs::class_derivative_validation_checks_the_finite_difference_against_the_regression_slope",
]
DERIVATIVE_TEMPORAL = [
    T
    + "temporal_admg_derivative.rs::every_derivative_functional_on_a_temporal_admg_publishes_a_consumable_artifact",
    T
    + "temporal_admg_derivative.rs::the_window_adjustment_recovers_the_structural_slope_with_a_block_band",
    T
    + "temporal_admg_derivative.rs::the_bayesian_derivative_publishes_a_credible_band_around_the_same_slope",
]
CP = T + "class_attribution_route.rs::"
GP = T + "graph_posterior_attribution_routes.rs::"
PS = T + "path_specific_admg_route.rs::"
PP = T + "path_specific_pag_route.rs::"
MM = T + "mediation_mixed_graph.rs::"
MV = T + "mediation_decomposition_validation.rs::"
CF = T + "counterfactual_class_route.rs::"
TA = T + "temporal_admg_route_evidence.rs::"
CD = T + "class_and_design_validation_suites.rs::"
# The tests that run each closed route and encode its contracted result for an independent reader.
CLOSED_EVIDENCE = {
    ("AnomalyAttribution", "Admg"): [
        CP + "an_admg_anomaly_is_attributed_to_the_noise_blocks",
        ARTIFACTS + "static_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("AnomalyAttribution", "Cpdag"): [
        CP + "a_cpdag_anomaly_is_the_mean_of_its_dag_completions",
        CP + "a_bayesian_cpdag_publishes_a_shared_posterior_and_never_a_mixture",
        ARTIFACTS + "static_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("AnomalyAttribution", "Pag"): [
        CP + "a_pag_with_a_hedged_completion_keeps_the_identified_mass",
        ARTIFACTS + "static_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("AnomalyAttribution", "TemporalCpdag"): [
        CP + "a_temporal_cpdag_is_the_mean_of_its_temporal_dag_completions",
        ARTIFACTS + "temporal_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("AnomalyAttribution", "TemporalPag"): [
        ARTIFACTS + "temporal_class_attributions_publish_an_artifact_a_consumer_reads"
    ],
    ("AnomalyAttribution", "TemporalAdmg"): [
        CP + "a_temporal_admg_whose_latent_edge_misses_the_ancestors_is_the_temporal_dag_route",
        ARTIFACTS + "temporal_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("ChangeAttribution", "Admg"): [
        CP + "an_admg_change_matches_the_known_shift_of_each_block",
        ARTIFACTS + "static_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("ChangeAttribution", "Cpdag"): [
        CP + "a_cpdag_change_that_every_member_agrees_on_keeps_one_value",
        ARTIFACTS + "static_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("ChangeAttribution", "Pag"): [
        GP + "a_change_attribution_carries_the_graph_sensitivity_across_the_atoms",
        ARTIFACTS + "static_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("ChangeAttribution", "TemporalCpdag"): [
        CP + "a_temporal_cpdag_is_the_mean_of_its_temporal_dag_completions",
        ARTIFACTS + "temporal_class_attributions_publish_an_artifact_a_consumer_reads",
    ],
    ("ChangeAttribution", "TemporalPag"): [
        ARTIFACTS + "temporal_class_attributions_publish_an_artifact_a_consumer_reads"
    ],
    ("ChangeAttribution", "TemporalAdmg"): [
        ARTIFACTS + "temporal_class_attributions_publish_an_artifact_a_consumer_reads"
    ],
    ("Counterfactual", "Admg"): [
        CF + "an_admg_counterfactual_is_the_enumerated_effect_on_the_treated",
        CF + "the_bayesian_bootstrap_posterior_centres_on_the_identified_functional",
        CF + "every_class_counterfactual_publishes_an_artifact_a_consumer_reads",
    ],
    ("Counterfactual", "Cpdag"): [
        CF + "a_cpdag_whose_completions_disagree_publishes_the_mean_and_the_identified_set",
        CF + "every_class_counterfactual_publishes_an_artifact_a_consumer_reads",
    ],
    ("Counterfactual", "Pag"): [
        CF + "a_pag_is_read_through_the_admgs_its_mag_stands_for",
        CF + "every_class_counterfactual_publishes_an_artifact_a_consumer_reads",
    ],
    ("Counterfactual", "TemporalAdmg"): [
        CF + "a_temporal_admg_reads_its_contemporaneous_projection_and_resamples_in_blocks",
        CF + "every_class_counterfactual_publishes_an_artifact_a_consumer_reads",
    ],
    ("MediationEffect", "Admg"): [
        MM + "an_admg_decomposition_is_the_counted_total_direct_and_indirect_effect",
        MM + "the_bayesian_bootstrap_pairs_the_component_posteriors",
        MM + "mixed_graph_mediation_and_distribution_publish_an_artifact_a_consumer_reads",
    ],
    ("MediationEffect", "Cpdag"): [
        MM
        + "the_contemporaneous_projection_and_a_fully_oriented_cpdag_give_the_same_decomposition",
        MM + "mixed_graph_mediation_and_distribution_publish_an_artifact_a_consumer_reads",
    ],
    ("MediationEffect", "Pag"): [
        MV + "a_pag_decomposition_is_refuted_member_by_member_against_the_structural_effect",
        MV + "a_pag_decomposition_publishes_an_artifact_a_consumer_reads",
    ],
    ("MediationEffect", "TemporalAdmg"): [
        MM + "a_temporal_admg_window_with_one_period_reproduces_the_counted_decomposition",
        MM + "mixed_graph_mediation_and_distribution_publish_an_artifact_a_consumer_reads",
    ],
    ("TemporalMediationEffect", "TemporalPag"): [
        T
        + "temporal_pag_mediation.rs::a_lagged_window_matches_the_structural_decomposition_of_the_series",
        T
        + "temporal_pag_mediation.rs::a_temporal_pag_decomposition_publishes_an_artifact_a_consumer_reads",
    ],
    ("TemporalMediationEffect", "TemporalAdmg"): [
        T
        + "temporal_pag_mediation.rs::a_two_period_temporal_admg_decomposition_publishes_an_artifact_a_consumer_reads",
        MM + "a_temporal_admg_window_with_one_period_reproduces_the_counted_decomposition",
        MM + "mixed_graph_mediation_and_distribution_publish_an_artifact_a_consumer_reads",
        T
        + "temporal_admg_posterior.rs::a_mediation_posterior_mixes_each_component_over_the_atoms_that_identify_it",
    ],
    ("NestedCounterfactualEffect", "Admg"): [
        PS + "natural_direct_effect_with_covariate_confounding_matches_the_structural_effect",
        PS + "contracted_results_round_trip_through_the_consumer",
    ],
    ("NestedCounterfactualEffect", "Cpdag"): [
        PS + "the_natural_direct_effect_over_a_class_matches_its_one_completion",
        PS + "contracted_results_round_trip_through_the_consumer",
    ],
    ("NestedCounterfactualEffect", "Pag"): [
        PP + "the_natural_direct_effect_over_a_pag_lies_in_its_identified_set",
        PP + "contracted_pag_results_round_trip_through_the_consumer",
    ],
    ("NestedCounterfactualEffect", "TemporalAdmg"): [
        PS + "contracted_results_round_trip_through_the_consumer",
        PS + "prepared_handles_run_through_the_estimator_route_and_equal_a_fresh_run",
    ],
    ("PathSpecificEffect", "Admg"): [
        PS + "effect_through_a_confounded_child_uses_the_district_formula",
        PS + "a_bootstrap_interval_accompanies_the_frequentist_point",
        PS + "contracted_results_round_trip_through_the_consumer",
    ],
    ("PathSpecificEffect", "Cpdag"): [
        PS + "a_class_with_one_completion_reproduces_the_structural_effect",
        PS + "completions_that_disagree_publish_the_identified_set",
        PS + "contracted_results_round_trip_through_the_consumer",
    ],
    ("PathSpecificEffect", "Pag"): [
        PP + "a_pag_with_visible_edges_reproduces_the_structural_effect",
        PP + "a_bootstrap_interval_accompanies_the_pag_point",
        PP + "contracted_pag_results_round_trip_through_the_consumer",
    ],
    ("PathSpecificEffect", "TemporalAdmg"): [
        PS + "contracted_results_round_trip_through_the_consumer",
        PS + "prepared_handles_run_through_the_estimator_route_and_equal_a_fresh_run",
    ],
    ("InterventionalDistribution", "Cpdag"): [
        MM + "a_cpdag_whose_completions_agree_gives_the_counted_interventional_distribution",
        MM + "completions_that_disagree_publish_the_identified_set_of_each_atom",
        MM + "mixed_graph_mediation_and_distribution_publish_an_artifact_a_consumer_reads",
    ],
    ("InterventionalDistribution", "Pag"): [
        MV + "a_pag_interventional_distribution_publishes_an_artifact_a_consumer_reads"
    ],
    ("PulseEffect", "TemporalAdmg"): [
        TA + "a_pulse_is_the_counted_contrast_and_stays_unlicensed",
        TA + "overlapping_windows_are_resampled_in_blocks_and_never_row_by_row",
        TA + "the_bayesian_form_shares_one_dirichlet_weight_per_block",
    ],
    ("SustainedEffect", "TemporalAdmg"): [
        TA + "a_sustained_contrast_sets_every_treated_period",
        TA + "overlapping_windows_are_resampled_in_blocks_and_never_row_by_row",
    ],
    ("TemporalConditionalPulse", "TemporalAdmg"): [
        TA + "a_conditional_contrast_is_idc_on_the_same_window",
        T
        + "admg_conditional_effect.rs::a_temporal_admg_conditional_is_the_contemporaneous_projection_contrast",
    ],
    ("TemporalConditionalSustained", "TemporalAdmg"): [
        TA + "a_conditional_sustained_contrast_is_idc_on_the_same_window"
    ],
    ("ConditionalEffect", "TemporalAdmg"): [
        T
        + "admg_conditional_effect.rs::a_temporal_admg_conditional_is_the_contemporaneous_projection_contrast",
        T
        + "graph_posterior_functional_routes.rs::the_temporal_admg_conditional_mixture_is_the_counted_contrast_with_the_lost_mass",
        T
        + "graph_posterior_functional_routes.rs::the_bayesian_temporal_admg_conditional_pools_the_atoms_draw_by_draw",
    ],
    ("TransportQuery", "Dag"): [
        T
        + "checked_transport_trial_route_evidence.rs::a_dag_is_the_confounder_free_selection_diagram_of_a_transport_query",
        T
        + "checked_transport_trial_route_evidence.rs::transport_trial_executes_from_retained_sid_proof_after_builder_drop",
    ],
    ("TransportQuery", "Cpdag"): [
        CD + "a_cpdag_and_a_pag_transport_publish_an_artifact_a_consumer_reads",
        CD + "a_cpdag_and_a_pag_run_the_transport_suite_on_both_inferences",
        T
        + "graph_posterior_exposure_routes.rs::a_cpdag_whose_members_certify_different_formulas_is_partially_identified",
    ],
    ("TransportQuery", "Pag"): [
        CD + "a_cpdag_and_a_pag_transport_publish_an_artifact_a_consumer_reads",
        CD + "a_cpdag_and_a_pag_run_the_transport_suite_on_both_inferences",
        T
        + "graph_posterior_exposure_routes.rs::a_pag_is_read_through_the_graphs_its_completions_stand_for",
    ],
}
for _class in ("Admg", "Cpdag", "Pag", "TemporalAdmg"):
    CLOSED_EVIDENCE[("InterferenceQuery", _class)] = [
        CD + "every_class_interference_route_publishes_an_artifact_a_consumer_reads",
        T
        + "staged_attribution_transport_interference.rs::interference_is_stated_under_every_static_structure_class_with_the_design_identification",
        CD
        + "supplied_exposure_probabilities_are_checked_against_the_realized_exposures_on_every_class",
    ]
    for _q in (
        "AverageDerivative",
        "PointDerivative",
        "DirectionalDerivative",
        "Elasticity",
        "SemiElasticity",
        "ResponseJacobian",
    ):
        CLOSED_EVIDENCE[(_q, _class)] = list(
            DERIVATIVE_TEMPORAL if _class == "TemporalAdmg" else DERIVATIVE_STATIC
        )
for (_q, _class), _tests in CLOSED_EVIDENCE.items():
    _rid = f"closed.{_q}.{_class}"
    EVIDENCE_OVERRIDE[_rid] = _tests
    DECLARED_KIND[_rid] = "design_aware_resampling"
    IMPLEMENTATION_EXTRA[_rid] = CLOSED_IMPLEMENTATION[_q]


def sources_of(ev: Evidence, m: dict) -> list[str]:
    owner = m["owning_module"]
    files = [owner] if (ev.root / owner).is_file() else []
    files.extend(implementation_extra(m["route"]))
    folder = (ev.root / owner).parent
    stem = Path(owner).stem
    for sibling in sorted(folder.glob(f"{stem}*.rs")) + sorted(
        (folder / stem).glob("*.rs") if (folder / stem).is_dir() else []
    ):
        files.append(sibling.relative_to(ev.root).as_posix())
    return list(dict.fromkeys(files))


def current_text(ev: Evidence) -> str:
    text = ev.text(MANIFEST)
    return text.split(MARK)[0].rstrip("\n") + "\n\n" if MARK in text else text.rstrip("\n") + "\n\n"


def report(methods: list[dict]) -> str:
    lines = [f"methods: {len(methods)}"]
    lines.append("kinds: " + json.dumps(Counter(m["kind"] for m in methods)))
    lines.append("uncertainty: " + json.dumps(Counter(m["uncertainty"] for m in methods)))
    lines.append(
        "basis: "
        + json.dumps(Counter(m["uncertainty_basis"].split("|")[0].split(":")[0] for m in methods))
    )
    lines.append("inference: " + json.dumps(Counter(m["inference"] for m in methods)))
    return "\n".join(lines)


def self_test() -> int:
    text = (
        "pub(super) const fn identifier_data(id: IdentifierId) -> IdentifierData {\n    match id {\n"
        '        IdentifierId::BackdoorAdjustment => IdentifierData {\n            name: "backdoor.adjustment",\n        },\n    }\n}\n'
        "const ADJUSTMENT: &[IdentifierId] = &[\n    IdentifierId::BackdoorAdjustment,\n];\n"
        "pub(super) const fn estimator_data(id: EstimatorId) -> EstimatorData {\n    match id {\n"
        '        EstimatorId::Aipw => EstimatorData {\n            compatible: ADJUSTMENT,\n            name: "aipw",\n'
        '            provenance: same_step("estimate.aipw"),\n        },\n'
        '        EstimatorId::GlmAdjustment => EstimatorData {\n            compatible: ADJUSTMENT,\n            name: "glm.adjustment",\n'
        '            provenance: ("estimate.glm_adjustment", "estimate.glm_adjustment_ate"),\n        },\n    }\n}\n'
    )
    table = parse_estimators(text)
    assert table["aipw"] == {
        "variant": "Aipw",
        "identifiers": ["backdoor.adjustment"],
        "provenance": ("estimate.aipw", "estimate.aipw"),
    }, table
    assert table["glm.adjustment"]["provenance"] == (
        "estimate.glm_adjustment",
        "estimate.glm_adjustment_ate",
    )
    sides = inference_classes(
        'const REFUSED_UNDER_BAYESIAN: &[&str] = &[\n    "aipw",\n];\nconst REFUSED_UNDER_FREQUENTIST: &[&str] = &[\n    "bayesian.gcomp",\n];\n'
    )
    assert sides == {"aipw": "Frequentist", "bayesian.gcomp": "Bayesian"}, sides
    blocks = split_tests("x/tests/t.rs", "#[test]\nfn a() { one }\n#[test]\nfn b() { two }\n")
    assert (
        [n for n, _ in blocks] == ["a", "b"] and "two" in blocks[1][1] and "two" not in blocks[0][1]
    )
    pyblocks = split_tests(
        "python/tests/test_t.py",
        "def helper():\n    pass\ndef test_x():\n    one\ndef test_y():\n    two\n",
    )
    assert [n for n, _ in pyblocks] == ["test_x", "test_y"]
    assert design_of("TemporalDag") == "temporal" and design_of("Dag") == "static"
    sample = {
        f: (
            []
            if f
            in (
                "identifier",
                "lowering",
                "structures",
                "validations",
                "uncertainty_by_structure",
                "implementation",
                "provenance",
                "evidence",
                "artifact_consumer",
            )
            else 0
            if f == "coordinates"
            else "x"
        )
        for f in FIELDS
    }
    rendered = render([sample])
    assert (
        rendered.startswith(MARK)
        and "[[method]]" in rendered
        and tomllib.loads(rendered)["method"][0]["id"] == "x"
    )
    print("route_methods self-test passed")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    ev = Evidence(ROOT)
    src = Sources(ev)
    methods = build(src)
    text = current_text(ev) + render(methods) + "\n"
    path = ROOT / MANIFEST
    if "--report" in argv:
        print(report(methods))
        return 0
    if "--check" in argv:
        stale = path.read_text(encoding="utf-8") != text
        if stale:
            print(
                f"FAIL: {MANIFEST} method records are stale; run python3 scripts/route_methods.py"
            )
        problems = verify(src, methods)
        for line in problems:
            print(f"FAIL: {line}")
        print(f"route methods: {len(methods)} records, {len(problems)} finding(s)")
        return 1 if stale or problems else 0
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(methods)} method records to {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
