#!/usr/bin/env bash
# Scheduled statistical calibration gate (DESIGN.md §28.3).
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

echo "== SE analytic / bootstrap CI coverage (causal-estimate) =="
run_ignored causal-estimate linear_adjustment_analytic_ci_coverage
run_ignored causal-estimate linear_adjustment_hc1_ci_coverage
run_ignored causal-estimate ipw_hajek_bootstrap_ci_coverage
run_ignored causal-estimate ipw_hajek_analytic_ci_coverage
run_ignored causal-estimate aipw_analytic_ci_coverage
run_ignored causal-estimate matching_homoskedastic_ci_coverage
run_ignored causal-estimate wald_iv_analytic_ci_coverage
run_ignored causal-estimate wald_iv_hc1_ci_coverage

echo "== CI Type I / permutation uniformity (causal-stats) =="
run_ignored causal-stats robust_parcorr_calibration_gate
run_ignored causal-stats weighted_parcorr_calibration_gate
run_ignored causal-stats gsquared_calibration_gate
run_ignored causal-stats knn_cmi_calibration_gate
run_ignored causal-stats parcorr_perm_pvalue_uniformity_gate
run_ignored causal-stats knn_perm_pvalue_uniformity_gate

echo "== Discovery null FPR / power (causal-discovery) =="
run_ignored causal-discovery pc_null_fpr_near_alpha
run_ignored causal-discovery pcmci_null_fpr_near_alpha
run_ignored causal-discovery pcmci_planted_lag1_power

echo "== Discovery synthetic-null validator (causal-validate) =="
run_ignored causal-validate synthetic_null_fpr_near_alpha_gate

echo "gate_calibration: ok"
