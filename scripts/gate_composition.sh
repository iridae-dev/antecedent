#!/usr/bin/env bash
# 1.10 composition gate: exact consuming tests must run, not merely be cited.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== 1.10 composition consuming tests =="

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

run_and_count "antecedent v110_contract" \
  cargo test -p antecedent --test v110_contract -- --nocapture

run_and_count "antecedent prepared identify counts" \
  cargo test -p antecedent --test prepared_analysis prepared_second_shot_reuses_identification -- --nocapture

run_and_count "antecedent-io identity encoding" \
  cargo test -p antecedent-io --lib encoding_rule_changes_change_the_advertised_digest -- --nocapture

run_and_count "antecedent-io dbn atom identity" \
  cargo test -p antecedent-io --lib dbn_atom_identity_includes_lags_namespace_and_execution_key -- --nocapture

if [[ "${SKIP_PYTHON_SMOKE:-0}" == "1" ]]; then
  echo "SKIP_PYTHON_SMOKE=1; skipping Python v110 contract tests"
elif ! command -v uv >/dev/null 2>&1; then
  echo "WARN: uv not on PATH; skipping Python v110 contract tests"
else
  (
    cd python
    log="$(mktemp)"
    if ! uv run pytest -q tests/test_v110_contract.py >"$log" 2>&1; then
      echo "FAIL: python test_v110_contract"
      cat "$log"
      rm -f "$log"
      exit 1
    fi
    ran="$(grep -cE 'passed' "$log" || true)"
    if [[ "$ran" -lt 1 ]]; then
      echo "FAIL: python test_v110_contract reported no executed tests"
      cat "$log"
      rm -f "$log"
      exit 1
    fi
    echo "ok: python test_v110_contract"
    rm -f "$log"
  )
fi

echo "gate_composition: ok"
