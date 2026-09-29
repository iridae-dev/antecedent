#!/usr/bin/env bash
# 2.2 promotion gate: parity/promotion_2_2.toml freezes each 2.2 cell before
# implementation, keeps its routes closed until promotion, and rejects a
# capability that executes without its proof, refusal and artifact evidence.
#
#   bash scripts/gate_promotion.sh              # static contract, then execute cited fixtures
#   bash scripts/gate_promotion.sh --self-test  # broken registries must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REGISTRY=parity/promotion_2_2.toml

self_test() {
  local tmp status=0
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  if ! python3 scripts/check_promotion_records.py "$REGISTRY" >/dev/null; then
    echo "SELF-TEST FAIL: the committed registry was rejected"; status=1
  fi
  # Each narrow corruption, alone, must fail the static contract.
  # label|exact text|replacement (first occurrence only; empty deletes it).
  local -a cases=(
    'route licensed before promotion|status = "closed", reason_code = "cell_not_licensed"|status = "licensed"'
    'promoted without evidence|status = "frozen"|status = "promoted"'
    'nominal interval claim|inference_claim = "point_only"|inference_claim = "nominal"'
    'unregistered refusal code|code = "transport_missing_evidence"|code = "no_such_code"'
    'search without depth limit|depth_limit = "SearchLimits.depth", |'
    'search off the shared contract|contract = "antecedent_core::SearchBudget"|contract = "ZTransportLimits"'
    'missing frozen theorem|theorem = "mz-transportability|theorem_draft = "mz-transportability'
    'missing negative fixture|{ id = "x9.missing_joint.negative", role = "negative"|# '
    'calibrated without records|coverage_records = ["cov.classical|coverage_records = [] # ["cov.classical'
    'unnamespaced refusal detail|detail = "mz_transport.search_incomplete"|detail = "search incomplete"'
    'uncancellable bound|bounds = { horizon = 2, max_actions = 8, max_history_states = 4096, cancellation = true|bounds = { horizon = 2, max_actions = 8, max_history_states = 4096, cancellation = false'
  )
  local spec label old new
  for spec in "${cases[@]}"; do
    IFS='|' read -r label old new <<<"$spec"
    OLD="$old" NEW="$new" python3 -c 'import os, sys; t = sys.stdin.read(); sys.stdout.write(t.replace(os.environ["OLD"], os.environ["NEW"], 1))' \
      <"$REGISTRY" >"$tmp/case.toml"
    if cmp -s "$REGISTRY" "$tmp/case.toml"; then
      echo "SELF-TEST FAIL: corruption '$label' changed nothing"; status=1
    elif python3 scripts/check_promotion_records.py "$tmp/case.toml" >/dev/null 2>&1; then
      echo "SELF-TEST FAIL: '$label' passed the gate"; status=1
    else
      echo "self-test ok: '$label' fails"
    fi
  done
  # A route licensed in its owning registry ahead of its record must fail.
  cp parity/transport_stages.toml "$tmp/stages.toml"
  printf '\n[[routes]]\nroute = "antecedent_identify.mixed_source_search"\nstage = "identify"\nstatus = "licensed"\n' >>"$tmp/stages.toml"
  if PROMOTION_TRANSPORT_STAGES="$tmp/stages.toml" python3 scripts/check_promotion_records.py >/dev/null 2>&1; then
    echo "SELF-TEST FAIL: 'transport stage licensed before promotion' passed the gate"; status=1
  else
    echo "self-test ok: 'transport stage licensed before promotion' fails"
  fi
  cp parity/support_licensed.toml "$tmp/support.toml"
  printf '\n[[cell]]\nquery = "NestedCounterfactualEffect"\ncontrast = "natural_indirect"\n' >>"$tmp/support.toml"
  if PROMOTION_SUPPORT_LICENSED="$tmp/support.toml" python3 scripts/check_promotion_records.py >/dev/null 2>&1; then
    echo "SELF-TEST FAIL: 'support cell licensed before promotion' passed the gate"; status=1
  else
    echo "self-test ok: 'support cell licensed before promotion' fails"
  fi
  [[ "$status" -eq 0 ]] || return 1
  echo "gate_promotion self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

echo "== 2.2 promotion records =="
EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 scripts/check_promotion_records.py --emit-evidence "$EVIDENCE"

# Cited fixtures are executed, not just resolved: a promoted record's evidence
# must pass. Frozen records cite nothing yet, so there may be nothing to run.
if [[ -s "$EVIDENCE" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "FAIL: uv is required; unexecuted Python rows are not promotion evidence"
    exit 1
  fi
  echo "== promotion fixtures: executing cited evidence =="
  python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_promotion
else
  echo "no promotion fixture cites evidence yet; nothing to execute"
fi
echo "gate_promotion: ok"
