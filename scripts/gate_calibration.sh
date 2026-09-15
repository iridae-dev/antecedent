#!/usr/bin/env bash
# Scheduled statistical calibration gate.
# Not part of every-PR unit CI — run locally / before release / weekly GHA.
#
# Every group runs even when an earlier one fails; the failed groups are listed
# at the end and the script exits nonzero.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAILED=""
FAILED_COUNT=0
RECHECKED=""
GROUP_INDEX=0

# Sharding for the scheduled workflow: `ANTECEDENT_CALIBRATION_SHARD=k/N` runs
# only the groups whose index (in script order) is congruent to k modulo N, so
# N runners split the gate and each stays under its timeout. Unset runs all.
# `ANTECEDENT_CALIBRATION_DRY_RUN=1` lists the groups this shard would run.
SHARD_K=""
SHARD_N=""
if [ -n "${ANTECEDENT_CALIBRATION_SHARD:-}" ]; then
  SHARD_K="${ANTECEDENT_CALIBRATION_SHARD%%/*}"
  SHARD_N="${ANTECEDENT_CALIBRATION_SHARD##*/}"
  case "$SHARD_K$SHARD_N" in *[!0-9]*|"") echo "bad ANTECEDENT_CALIBRATION_SHARD=$ANTECEDENT_CALIBRATION_SHARD (want k/N)" >&2; exit 2;; esac
  if [ "$SHARD_N" -lt 1 ] || [ "$SHARD_K" -ge "$SHARD_N" ]; then
    echo "bad ANTECEDENT_CALIBRATION_SHARD=$ANTECEDENT_CALIBRATION_SHARD (want 0 <= k < N)" >&2; exit 2
  fi
fi

# Replicate count of the precision recheck (crates/antecedent/tests/common/calibration.rs).
RECHECK_NSIM="${ANTECEDENT_CALIBRATION_RECHECK_NSIM:-2000}"

# Run one gate group; record it as failed instead of aborting the gate.
#
# A coverage cell that passes its 400-replicate band but lands more than 2
# points under its level prints a `calibration-recheck` line. The group is then
# re-run at RECHECK_NSIM replicates, where the harness also enforces the
# one-sided precision floor (level − 2·MCSE), and that run's verdict stands.
check() {
  local label="$1"
  shift
  GROUP_INDEX=$((GROUP_INDEX + 1))
  if [ -n "$SHARD_N" ] && [ $(( (GROUP_INDEX - 1) % SHARD_N )) -ne "$SHARD_K" ]; then
    return 0
  fi
  if [ -n "${ANTECEDENT_CALIBRATION_DRY_RUN:-}" ]; then
    echo "group ${GROUP_INDEX}: ${label}"
    return 0
  fi
  local log status
  log="$(mktemp -t gate_calibration.XXXXXX)"
  "$@" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [ "$status" -eq 0 ] && grep -q '^calibration-recheck ' "$log"; then
    echo "== recheck at ${RECHECK_NSIM} replicates: ${label} =="
    RECHECKED="${RECHECKED}  ${label}"$'\n'
    ANTECEDENT_CALIBRATION_NSIM="$RECHECK_NSIM" "$@"
    status=$?
  fi
  rm -f "$log"
  if [ "$status" -ne 0 ]; then
    FAILED="${FAILED}  ${label}"$'\n'
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
}

# Coverage tests run 400 replicates each (two-sided level ± 3·MCSE band, then
# the precision recheck above), so the gate builds in release; a debug build is
# ~50x slower with identical numbers.
run_ignored() {
  local pkg="$1"
  local filter="$2"
  echo "== ${pkg}: ${filter} =="
  check "${pkg}: ${filter}" cargo test --release -p "$pkg" --lib "$filter" -- --ignored --nocapture
}

echo "== SE analytic / bootstrap CI coverage (antecedent-estimate) =="
# Two-sided 0.95 ± 3·MCSE band (calibration_coverage.rs), no floor or exemption.
run_ignored antecedent-estimate linear_adjustment_analytic_ci_coverage
run_ignored antecedent-estimate linear_adjustment_hc1_ci_coverage
run_ignored antecedent-estimate ipw_hajek_bootstrap_ci_coverage
run_ignored antecedent-estimate ipw_hajek_analytic_ci_coverage
run_ignored antecedent-estimate ipw_hajek_analytic_conformance_scm_ci_coverage
run_ignored antecedent-estimate aipw_analytic_ci_coverage
run_ignored antecedent-estimate aipw_ate_hc1_ci_coverage
run_ignored antecedent-estimate aipw_att_hc1_ci_coverage
run_ignored antecedent-estimate aipw_atc_hc1_ci_coverage
run_ignored antecedent-estimate aipw_att_cluster_ci_coverage
run_ignored antecedent-estimate matching_homoskedastic_ci_coverage
run_ignored antecedent-estimate wald_iv_analytic_ci_coverage
run_ignored antecedent-estimate wald_iv_hc1_ci_coverage
run_ignored antecedent-estimate iv_2sls_analytic_ci_coverage
run_ignored antecedent-estimate iv_2sls_hc1_heteroskedastic_ci_coverage
run_ignored antecedent-estimate frontdoor_stacked_hc0_ci_coverage
run_ignored antecedent-estimate frontdoor_stacked_hc1_ci_coverage
run_ignored antecedent-estimate rd_sharp_analytic_ci_coverage
run_ignored antecedent-estimate rd_sharp_hc1_heteroskedastic_ci_coverage

run_ignored antecedent-estimate bayesian_pulse_conjugate_nominal_90_coverage
run_ignored antecedent-estimate bayesian_sustained_single_step_conjugate_nominal_90_coverage
run_ignored antecedent-estimate bayesian_sustained_multi_step_conjugate_nominal_90_coverage
run_ignored antecedent-estimate bayesian_panel_hierarchical_nominal_90_coverage

echo "== 1.9 temporal / mixture interval coverage (antecedent) =="
run_ignored_test() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_calibration: ${filter}" \
    cargo test --release -p antecedent --test v19_calibration "$filter" -- --ignored --exact --nocapture
}
run_ignored_test frequentist_dbn_pulse_shared_block_nominal_90_coverage
run_ignored_test frequentist_dbn_sustained_shared_block_nominal_90_coverage
run_ignored_test frequentist_dbn_multistep_sustained_shared_block_nominal_90_coverage
run_ignored_test frequentist_dbn_pulse_single_atom_baseline_nominal_90_coverage
run_ignored_test frequentist_dbn_pulse_ar1_rho05_n160_nominal_90_coverage
# Boundary cell: measured 0.879 at 2000 replicates (mixture centred on the atoms' limits).
run_ignored_test frequentist_dbn_pulse_ar1_rho09_n400_boundary_within_band
run_ignored_test frequentist_dbn_pulse_ar1_rho05_n60_nominal_90_coverage
run_ignored_test frequentist_dbn_multistep_ar1_rho05_n160_nominal_90_coverage
run_ignored_test frequentist_dbn_multistep_ar1_rho09_n400_nominal_90_coverage
run_ignored_test frequentist_dbn_multistep_ar1_rho05_n60_nominal_90_coverage
run_ignored_test frequentist_temporal_cpdag_pulse_envelope_nominal_90_coverage
run_ignored_test frequentist_temporal_cpdag_sustained_envelope_nominal_90_coverage
run_ignored_test frequentist_temporal_cpdag_pulse_ar1_rho05_n160_nominal_90_coverage
# Boundary cell: asserted against the band around its measured coverage.
run_ignored_test frequentist_temporal_cpdag_pulse_ar1_rho09_n400_boundary_within_band
run_ignored_test frequentist_temporal_cpdag_pulse_ar1_rho05_n60_nominal_90_coverage
run_ignored_test frequentist_temporal_pag_pulse_envelope_nominal_90_coverage
run_ignored_test bayesian_temporal_dag_pulse_staged_nominal_90_coverage
run_ignored_test bayesian_temporal_dag_multistep_sustained_nominal_90_coverage
run_ignored_test bayesian_temporal_dag_response_curve_pointwise_band_coverage
run_ignored_test bayesian_temporal_cpdag_pulse_class_prior_nominal_90_coverage
run_ignored_test bayesian_temporal_cpdag_sustained_class_prior_nominal_90_coverage
run_ignored_test class_prior_mixture_functional_nominal_90_coverage
run_ignored_test bayesian_temporal_pag_pulse_nominal_90_coverage
run_ignored_test bayesian_dbn_posterior_pulse_nominal_90_coverage
run_ignored_test bayesian_temporal_cpdag_mediation_envelope_nominal_90_coverage
run_ignored_test bayesian_temporal_cpdag_mediation_unconfounded_nominal_90_coverage
run_ignored_test bayesian_temporal_dag_mediation_confounded_nominal_90_coverage

echo "== 1.9 temporal class envelopes: multi-step, TemporalPag, identified-set intervals (antecedent) =="
run_temporal_class() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_temporal_class_calibration: ${filter}" \
    cargo test --release -p antecedent --test v19_temporal_class_calibration "$filter" \
    -- --ignored --exact --nocapture
}
run_temporal_class frequentist_temporal_cpdag_multistep_sustained_nominal_90_coverage
run_temporal_class frequentist_temporal_cpdag_multistep_sustained_ar1_rho05_n160_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_multistep_sustained_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_multi_completion_pulse_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_sustained_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_pulse_ar1_rho05_n160_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_pulse_ar1_rho05_n60_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_sustained_ar1_rho05_n160_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_sustained_ar1_rho05_n60_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_one_completion_sustained_ar1_rho05_n160_nominal_90_coverage
# Below the mixture short-series threshold: warning enforced on >= 95% of
# replicates, coverage recorded, not gated.
run_temporal_class frequentist_temporal_pag_pulse_ar1_rho09_n400_short_series_boundary
run_temporal_class frequentist_temporal_pag_sustained_ar1_rho09_n400_short_series_boundary
run_temporal_class frequentist_temporal_cpdag_pulse_ar1_rho095_n160_short_series_boundary
run_temporal_class bayesian_temporal_pag_sustained_class_prior_nominal_90_coverage
run_temporal_class bayesian_temporal_pag_pulse_class_prior_ar1_rho05_n160_nominal_90_coverage
run_temporal_class bayesian_temporal_pag_sustained_class_prior_ar1_rho05_n160_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_identified_set_interval_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_multistep_identified_set_interval_nominal_90_coverage
run_temporal_class frequentist_temporal_cpdag_identified_set_interval_nominal_90_coverage
run_temporal_class frequentist_temporal_pag_point_identified_set_interval_nominal_90_coverage
run_temporal_class bayesian_temporal_pag_no_class_prior_identified_set_nominal_90_coverage
run_temporal_class bayesian_temporal_pag_sustained_no_class_prior_identified_set_nominal_90_coverage

echo "== 1.9 shared circular-block length sensitivity (x0.5 / x1 / x2 of the production length) =="
run_ignored antecedent analysis::execute::block_length_tests::shared_block_length_sensitivity

echo "== 1.9 static graph-posterior mixture coverage (antecedent) =="
run_static_mixture_test() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_static_mixture_calibration: ${filter}" \
    cargo test --release -p antecedent --test v19_static_mixture_calibration "$filter" \
    -- --ignored --nocapture
}
run_static_mixture_test static_graph_posterior_frequentist_ate_joint_if_nominal_90_coverage
run_static_mixture_test static_graph_posterior_frequentist_cate_joint_if_nominal_90_coverage
run_static_mixture_test static_graph_posterior_bayesian_ate_bma_nominal_90_coverage
run_static_mixture_test static_graph_posterior_bayesian_cate_bma_nominal_90_coverage

echo "== 1.9 derivative-family interval coverage (antecedent) =="
run_ignored_derivative() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_derivative_calibration: ${filter}" \
    cargo test --release -p antecedent --test v19_derivative_calibration "$filter" \
    -- --ignored --nocapture --exact
}
run_ignored_derivative ade_frequentist_gaussian_treatment_nominal_90_coverage
run_ignored_derivative ade_bayesian_gaussian_treatment_nominal_90_coverage
run_ignored_derivative ade_skewed_heteroskedastic_treatment_probe
run_ignored_derivative point_derivative_frequentist_curvature_nominal_90_coverage
run_ignored_derivative point_derivative_bayesian_curvature_nominal_90_coverage
run_ignored_derivative point_derivative_order_2_frequentist_curvature_nominal_90_coverage
run_ignored_derivative point_derivative_order_2_bayesian_curvature_nominal_90_coverage
run_ignored_derivative semi_elasticity_log_treatment_frequentist_nominal_90_coverage
run_ignored_derivative semi_elasticity_log_treatment_bayesian_nominal_90_coverage
run_ignored_derivative semi_elasticity_log_outcome_bayesian_nominal_90_coverage
run_ignored_derivative elasticity_bayesian_nominal_90_coverage
# Boundary cell: coordinate 0 of the GAM-gradient band measures 0.883 at 2000
# replicates on the gate's seeds; the band covers 0.892-0.897 over 10 000 designs.
run_ignored_derivative response_jacobian_bayesian_boundary_within_band
run_ignored_derivative directional_derivative_bayesian_nominal_90_coverage

echo "== 1.9 Bayesian temporal Pulse / Sustained under serial dependence (antecedent) =="
run_ignored_bayes_temporal() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_bayesian_temporal: ${filter}" \
    cargo test --release -p antecedent --test v19_bayesian_temporal "$filter" -- --ignored --nocapture
}
run_ignored_bayes_temporal bayesian_temporal_pulse_iid_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ar1_rho05_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ar1_rho09_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ar1_rho05_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_iid_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar1_rho05_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar1_rho09_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar1_rho05_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_iid_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar1_rho05_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar1_rho09_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar1_rho05_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_cpdag_class_prior_ar1_rho05_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_iid_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_ar1_rho05_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_ar1_rho09_n400_nominal_90_coverage
# Higher-order and non-autoregressive dependence: AR(2), ARMA(1,1) and MA(2) treatment
# and residual, n = 60–400.
run_ignored_bayes_temporal bayesian_temporal_pulse_ar2_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ar2_n100_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ar2_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ar2_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar2_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar2_n100_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar2_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_ar2_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar2_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar2_n100_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar2_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_ar2_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_arma11_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_arma11_n100_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_arma11_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_arma11_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_arma11_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_single_arma11_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_arma11_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_arma11_n100_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_arma11_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_sustained_multi_arma11_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ma2_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_pulse_ma2_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_ar2_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_ar2_n100_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_ar2_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_ar2_n400_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_arma11_n60_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_arma11_n160_nominal_90_coverage
run_ignored_bayes_temporal bayesian_temporal_mediation_arma11_n400_nominal_90_coverage

echo "== 1.9 dependence-honest Frequentist TemporalDag SEs (R-1, R-2) =="
run_temporal_frequentist() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_temporal_frequentist: ${filter}" \
    cargo test --release -p antecedent --test v19_temporal_frequentist "$filter" \
    -- --ignored --exact --nocapture
}
run_temporal_frequentist temporal_dag_pulse_iid_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_ar05_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_ar09_n400_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_ar05_n60_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_h2_iid_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_h2_ar05_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_h2_ar09_n400_nominal_90_coverage
# Boundary cell: measured 0.885 at 2000 replicates (replicate-SD variability at 58 rows).
run_temporal_frequentist temporal_dag_pulse_h2_ar05_n60_boundary_within_band
run_temporal_frequentist temporal_dag_sustained_iid_n160_nominal_90_coverage
# Boundary cell: measured 0.874–0.895 across seed streams at 2000 replicates.
run_temporal_frequentist temporal_dag_sustained_ar05_n160_boundary_within_band
run_temporal_frequentist temporal_dag_sustained_ar09_n400_nominal_90_coverage
run_temporal_frequentist temporal_dag_sustained_ar05_n60_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_iid_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_ar05_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_ar09_n400_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_ar05_n60_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_confounded_iid_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_ar09_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_ar09_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_pulse_ar09_n60_nominal_90_coverage
run_temporal_frequentist temporal_dag_mediation_ar09_n60_nominal_90_coverage
run_temporal_frequentist temporal_dag_multistep_sustained_ar05_n160_nominal_90_coverage
run_temporal_frequentist temporal_dag_multistep_sustained_ar09_n400_nominal_90_coverage
# Below the family short-series threshold: warning enforced on >= 95% of
# replicates, coverage recorded, not gated.
run_temporal_frequentist temporal_dag_pulse_ar1_treatment_rho09_n60_short_series_boundary
run_temporal_frequentist temporal_dag_mediation_ar1_treatment_rho09_n60_short_series_boundary
run_temporal_frequentist temporal_dag_multistep_sustained_ar1_treatment_rho095_n60_short_series_boundary

echo "== 1.9 static envelope / tier coverage (antecedent, release) =="
run_static_envelope() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_static_envelope_calibration: ${filter}" \
    cargo test --release -p antecedent --test v19_static_envelope_calibration "$filter" \
    -- --ignored --nocapture --exact
}
run_static_envelope static_cpdag_ate_envelope_frequentist_nominal_90_coverage
run_static_envelope static_cpdag_ate_envelope_bayesian_nominal_90_coverage
run_static_envelope static_pag_ate_envelope_frequentist_nominal_90_coverage
run_static_envelope static_pag_ate_envelope_bayesian_nominal_90_coverage
run_static_envelope conditional_effect_dag_frequentist_nominal_90_coverage
run_static_envelope conditional_effect_dag_frequentist_small_subgroup_nominal_90_coverage
run_static_envelope conditional_effect_dag_bayesian_nominal_90_coverage
run_static_envelope conditional_effect_cpdag_frequentist_nominal_90_coverage
run_static_envelope conditional_effect_cpdag_bayesian_nominal_90_coverage
run_static_envelope conditional_effect_pag_frequentist_nominal_90_coverage
run_static_envelope conditional_effect_pag_bayesian_nominal_90_coverage
run_static_envelope codetermined_aipw_closure_nominal_90_coverage
run_static_envelope unknown_two_scenario_joint_band_nominal_95_coverage

echo "== 1.9 temporal response surfaces: pointwise + simultaneous bands (antecedent) =="
# One invocation runs every ignored test in the file (Frequentist / Bayesian
# TemporalDag surfaces, observation-adjusted pairs, horizon-dependent I(h), and
# TemporalCpdag / TemporalPag completion atoms, two-step Sequence overlays on
# complete and observation-adjusted data, iid and AR(1) residuals).
check "v19_temporal_response_calibration" \
  cargo test --release -p antecedent --test v19_temporal_response_calibration -- --ignored --nocapture

echo "== 1.9 remaining static cells: responses, mediation, path, distribution, counterfactual (R-19, R-17) =="
run_static_remaining() {
  local filter="$1"
  echo "== antecedent: ${filter} =="
  check "v19_static_calibration: ${filter}" \
    cargo test --release -p antecedent --test v19_static_calibration "$filter" \
    -- --ignored --nocapture --exact
}
run_static_remaining response_curve_dag_frequentist_pointwise_nominal_90_coverage
run_static_remaining response_curve_dag_frequentist_simultaneous_nominal_90_coverage
run_static_remaining response_curve_dag_bayesian_pointwise_nominal_90_coverage
run_static_remaining intervention_response_dag_frequentist_nominal_90_coverage
run_static_remaining intervention_response_dag_bayesian_nominal_90_coverage
run_static_remaining intervention_response_cell_aipw_nominal_95_coverage
run_static_remaining class_response_cpdag_intervention_joint_if_nominal_90_coverage
run_static_remaining class_response_cpdag_curve_joint_if_pointwise_nominal_90_coverage
run_static_remaining mediation_nde_frequentist_nominal_90_coverage
run_static_remaining mediation_nie_frequentist_nominal_90_coverage
run_static_remaining mediation_nde_bayesian_nominal_90_coverage
run_static_remaining mediation_nie_bayesian_nominal_90_coverage
run_static_remaining path_specific_frequentist_nominal_90_coverage
run_static_remaining path_specific_bayesian_nominal_90_coverage
run_static_remaining path_specific_two_path_frequentist_nominal_90_coverage
run_static_remaining path_specific_two_path_bayesian_nominal_90_coverage
run_static_remaining interventional_distribution_bayesian_near_one_nominal_90_coverage
run_static_remaining interventional_distribution_bayesian_near_zero_nominal_90_coverage
run_static_remaining interventional_distribution_frequentist_near_one_nominal_95_coverage
run_static_remaining interventional_distribution_frequentist_near_zero_nominal_95_coverage
run_static_remaining interventional_distribution_frequentist_interior_nominal_95_coverage
run_static_remaining counterfactual_bayesian_mean_ite_nominal_90_coverage
# Gates the correctly specified law; the misspecified outcomes are recorded only.
run_static_remaining bayesian_gcomp_misspecification_probe
# Out-of-assumption probes: coverage recorded, not gated.
run_static_remaining response_curve_dag_frequentist_weak_overlap_probe
run_static_remaining mediation_confounded_mediator_probe

echo "== Bayesian posterior calibration (antecedent-validate) =="
run_ignored antecedent-validate \
  bayesian_checks::tests::calibration_gate::sbc_conjugate_gaussian_ranks_are_uniform
run_ignored antecedent-validate \
  bayesian_checks::tests::calibration_gate::posterior_calibration_synthetic_scm_nominal_90_coverage

echo "== CI Type I / permutation uniformity (antecedent-stats) =="
run_ignored antecedent-stats robust_parcorr_calibration_gate
run_ignored antecedent-stats weighted_parcorr_calibration_gate
run_ignored antecedent-stats gsquared_calibration_gate
run_ignored antecedent-stats knn_dependence_calibration_gate
run_ignored antecedent-stats parcorr_perm_pvalue_uniformity_gate
run_ignored antecedent-stats knn_perm_pvalue_uniformity_gate
run_ignored antecedent-stats multivariate_block_calibration_gate
run_ignored antecedent-stats multivariate_block_shuffle_calibration_gate
run_ignored antecedent-stats gpdc_block_shuffle_autocorrelated_type_i_gate
run_ignored antecedent-stats knn_unconditional_block_shuffle_autocorrelated_type_i_gate

echo "== Discovery null FPR / power (antecedent-discovery) =="
run_ignored antecedent-discovery pc_null_fpr_near_alpha
run_ignored antecedent-discovery pcmci_null_fpr_near_alpha
run_ignored antecedent-discovery pcmci_planted_lag1_power

echo "== 0.5.0 response/observation/transport/interference =="
check "gate_response_calibration.sh" bash scripts/gate_response_calibration.sh

if [ -n "$RECHECKED" ]; then
  echo "gate_calibration: group(s) rechecked at ${RECHECK_NSIM} replicates:"
  printf '%s' "$RECHECKED"
fi
if [ "$FAILED_COUNT" -gt 0 ]; then
  echo "gate_calibration: ${FAILED_COUNT} group(s) failed:"
  printf '%s' "$FAILED"
  exit 1
fi
if [ -n "$SHARD_N" ]; then
  echo "gate_calibration: shard ${SHARD_K}/${SHARD_N} ok (${GROUP_INDEX} groups in script order)"
else
  echo "gate_calibration: ok"
fi
