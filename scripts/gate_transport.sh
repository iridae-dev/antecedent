#!/usr/bin/env bash
# 2.0 transport gate: parity/transport_stages.toml drives every invocation.
#
# The registry contract (licensed routes resolve to collected tests; each T10
# fixture family has a positive and a counterexample) is checked statically,
# then every [[fixture_evidence]] row is executed by the shared row runner:
# Rust rows must pass exactly one test, Python rows at least one, none may fail.
# A missing uv/cargo runtime is a failure here, never a recorded pass.
#
#   bash scripts/gate_transport.sh              # full gate
#   bash scripts/gate_transport.sh --self-test  # broken registries must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REGISTRY=parity/transport_stages.toml

self_test() {
  local tmp status=0
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  if ! python3 scripts/check_transport_stages.py "$REGISTRY" >/dev/null; then
    echo "SELF-TEST FAIL: the committed registry was rejected"; status=1
  fi
  # Each narrow corruption, alone, must fail the static contract.
  local -a cases=(
    "missing counterexample|/^id = \"transport_budget_refusal.counterexample\"/,/^limits/d"
    "unknown assertion|s/exhausted_identification_budget_is_an_error_never_a_negative_witness/no_such_test/"
    "counterexample reuses positive|s/two_population_district_recursion_fails_truth_under_kernel_population_substitution/three_node_selection_graphs_agree_with_full_experimental_oracle/"
    "unlabelled evidence class|s/^evidence_class = \"theoretical_witness\"/evidence_class = \"proof\"/"
  )
  local spec label script
  for spec in "${cases[@]}"; do
    IFS='|' read -r label script <<<"$spec"
    sed -E "$script" "$REGISTRY" >"$tmp/case.toml"
    if cmp -s "$REGISTRY" "$tmp/case.toml"; then
      echo "SELF-TEST FAIL: corruption '$label' changed nothing"; status=1
    elif python3 scripts/check_transport_stages.py "$tmp/case.toml" >/dev/null 2>&1; then
      echo "SELF-TEST FAIL: '$label' passed the gate"; status=1
    else
      echo "self-test ok: '$label' fails"
    fi
  done
  [[ "$status" -eq 0 ]] || return 1
  echo "gate_transport self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required; unexecuted Python rows are not transport evidence"
  exit 1
fi

echo "== transport stage and fixture-family contracts =="
python3 scripts/check_transport_stages.py

echo "== T10 fixture families: positive and counterexample rows =="
python3 scripts/run_evidence_rows.py "$ROOT" "$REGISTRY" "$ROOT" fixture_evidence gate_transport

echo "== licensed stage routes: consuming evidence assertions =="
python3 scripts/run_evidence_rows.py "$ROOT" "$REGISTRY" "$ROOT" routes gate_transport

echo "revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "gate_transport: ok"
