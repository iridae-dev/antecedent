#!/usr/bin/env bash
# Scheduled statistical calibration gate.
# Not part of every-PR unit CI — run locally / before release / weekly GHA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_ignored() {
  local pkg="$1"
  local filter="$2"
  echo "== ${pkg}: ${filter} =="
  cargo test -p "$pkg" --lib "$filter" -- --ignored --nocapture
}

echo "== SE analytic / bootstrap CI coverage (antecedent-estimate) =="
run_ignored antecedent-estimate linear_adjustment_analytic_ci_coverage
run_ignored antecedent-estimate linear_adjustment_hc1_ci_coverage
run_ignored antecedent-estimate ipw_hajek_bootstrap_ci_coverage
run_ignored antecedent-estimate ipw_hajek_analytic_ci_coverage
run_ignored antecedent-estimate aipw_analytic_ci_coverage
run_ignored antecedent-estimate matching_homoskedastic_ci_coverage
run_ignored antecedent-estimate wald_iv_analytic_ci_coverage
run_ignored antecedent-estimate wald_iv_hc1_ci_coverage
run_ignored antecedent-estimate rd_sharp_analytic_ci_coverage

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
bash scripts/gate_response_calibration.sh

echo "gate_calibration: ok"
