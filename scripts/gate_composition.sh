#!/usr/bin/env bash
# 1.10 composition gate: exact consuming tests must run, not merely be cited.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== 1.10 composition consuming tests =="
REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
SKIPPED=()

echo "== existing evidence gates over composition records =="
bash scripts/gate_parity_schema.sh
bash scripts/gate_provenance_schema.sh
bash scripts/gate_metadata_consistency.sh
bash scripts/gate_evidence_reachability.sh

run_and_count() {
  local label="$1"
  shift
  local log
  log="$(mktemp)"
  if ! "$@" >"$log" 2>&1; then
    echo "FAIL: $label"
    cat "$log"
    rm -f "$log"
    exit 1
  fi
  local ran
  ran="$(grep -cE '^test .* \.\.\. ok$|^python/tests/.* PASSED$' "$log" || true)"
  if [[ "$ran" -lt 1 ]]; then
    echo "FAIL: $label reported no executed tests"
    cat "$log"
    rm -f "$log"
    exit 1
  fi
  echo "ok: $label ($ran tests)"
  rm -f "$log"
}

run_and_count "antecedent-io contract_section" \
  cargo test -p antecedent-io --lib contract_section -- --nocapture

run_and_count "antecedent-core request identity" \
  cargo test -p antecedent-core --lib request_identity_conflicts_on_reused_key -- --nocapture

run_and_count "antecedent-io validate_result seams" \
  cargo test -p antecedent-io --lib validate_result_ -- --nocapture

run_and_count "antecedent-core provenance ancestry" \
  cargo test -p antecedent-core --lib try_push_rejects_duplicates_and_cycles -- --nocapture

run_and_count "antecedent inspect does not identify" \
  cargo test -p antecedent --test v110_contract -- --nocapture inspect_does_not_identify

run_and_count "antecedent v110_contract" \
  cargo test -p antecedent --test v110_contract -- --nocapture

# New licensed-coordinate tests must keep the licensed_family_ prefix
# so this filter remains the composition enrollment list.
# 1.10 leftover B/C now included: Pulse/Sustained Bayes + cheap/full,
# static InterventionResponse, PathSpecific Bayes, TemporalCpdag/Pag,
# identifying TemporalPag Pulse/response, GP temporal Pulse encode,
# RightCensored / Interval / Truncated observation pairs, and
# prior-mapping / catalog transfer variants.
run_and_count "antecedent v110 licensed families" \
  cargo test -p antecedent --test v110_contract licensed_family_ -- --nocapture

run_and_count "antecedent v110 licensed compiler inspect" \
  cargo test -p antecedent --test v110_licensed_compiler -- --nocapture

run_and_count "antecedent panel response cluster bands" \
  cargo test -p antecedent --test data_modalities -- --nocapture \
  'panel_response_curve_uses_unit_cluster_bands|bayesian_panel_response_stays_refused|incomplete_class_panel_response_stays_refused'

run_and_count "antecedent panel class Pulse masses" \
  cargo test -p antecedent --test data_modalities -- --nocapture \
  'panel_class_pulse_uses_completion_masses|bayesian_panel_class_pulse_stays_refused'

run_and_count "antecedent prepared identify counts" \
  cargo test -p antecedent --test prepared_analysis prepared_second_shot_reuses_identification -- --nocapture

run_and_count "antecedent prepared family contracts" \
  cargo test -p antecedent --test prepared_analysis -- --nocapture \
  'prepared_reestimate_matches_fresh_analyze|prepared_conditional_effect_reestimate_matches_fresh|prepared_conditional_bayesian_records_bayesian_estimator|prepared_path_specific_reestimate_matches_fresh|prepared_distribution_reestimate_matches_fresh|prepare_accepts_temporal_effect_query_and_reuses_identification'

run_and_count "antecedent-io identity encoding" \
  cargo test -p antecedent-io --lib encoding_rule_changes_change_the_advertised_digest -- --nocapture

run_and_count "antecedent-io dbn atom identity" \
  cargo test -p antecedent-io --lib dbn_atom_identity_includes_lags_namespace_and_execution_key -- --nocapture

run_and_count "antecedent-io score reuse identity" \
  cargo test -p antecedent-io --lib score_reuse_keys_are_stricter_than_identification -- --nocapture

run_and_count "antecedent-state expected-version publish" \
  cargo test -p antecedent-state --lib refresh_results_at_rejects_a_stale_expected_version -- --nocapture

run_and_count "antecedent prepared series refresh atomicity" \
  cargo test -p antecedent --lib failed_series_refresh_leaves_handle_unchanged -- --nocapture

run_and_count "antecedent v110 series/state reuse" \
  cargo test -p antecedent --test v110_contract -- --nocapture \
  'failed_series_refresh_then_estimate_on_old_data|temporal_state_events_keep_lineage_and_refuse_stale_publish'

run_and_count "antecedent v15 score/batch reuse keys" \
  cargo test -p antecedent --test v15_numeric_pins -- --nocapture \
  'prepared_batch_shares_fold_object_and_covariate_design|replacement_data_changes_score_reuse_not_identification'

run_and_count "antecedent-core capability reports" \
  cargo test -p antecedent-core --lib capability -- --nocapture

run_and_count "antecedent support neighbors" \
  cargo test -p antecedent --lib licensed_neighbors_keep_graph_class_and_never_relabel -- --nocapture

run_and_count "antecedent v110 capability reports" \
  cargo test -p antecedent --test v110_contract -- --nocapture \
  capability_

run_and_count "antecedent-core claim handoffs" \
  cargo test -p antecedent-core --lib claim -- --nocapture

run_and_count "antecedent v110 claim handoffs" \
  cargo test -p antecedent --test v110_contract -- --nocapture \
  'claim_handoff_|derived_claim_'

run_and_count "antecedent v110 design ranking adapters" \
  cargo test -p antecedent --test v110_contract -- --nocapture \
  design_rank_

run_and_count "antecedent v110 consuming compositions" \
  cargo test -p antecedent --test v110_contract -- --nocapture \
  composition_

if [[ "${SKIP_PYTHON_SMOKE:-0}" == "1" ]]; then
  SKIPPED+=("python test_v110_contract (SKIP_PYTHON_SMOKE=1)")
  SKIPPED+=("python test_native_stub_conformance (SKIP_PYTHON_SMOKE=1)")
elif ! command -v uv >/dev/null 2>&1; then
  SKIPPED+=("python test_v110_contract (uv missing)")
  SKIPPED+=("python test_native_stub_conformance (uv missing)")
else
  (
    cd python
    log="$(mktemp)"
    if ! uv run pytest -q tests/test_v110_contract.py tests/test_native_stub_conformance.py >"$log" 2>&1; then
      echo "FAIL: python v110/stub conformance"
      cat "$log"
      rm -f "$log"
      exit 1
    fi
    ran="$(grep -cE 'passed' "$log" || true)"
    if [[ "$ran" -lt 1 ]]; then
      echo "FAIL: python v110/stub conformance reported no executed tests"
      cat "$log"
      rm -f "$log"
      exit 1
    fi
    echo "ok: python v110/stub conformance"
    rm -f "$log"
  )
fi

echo "revision: ${REVISION}"
if ((${#SKIPPED[@]})); then
  echo "skipped (not evidence):"
  for item in "${SKIPPED[@]}"; do
    echo "  - ${item}"
  done
  echo "gate_composition: rust consuming tests ran; skipped checks are not evidence"
else
  echo "gate_composition: ok"
fi
