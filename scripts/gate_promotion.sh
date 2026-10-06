#!/usr/bin/env bash
# 2.2 promotion gate: parity/promotion_2_2.toml freezes each 2.2 cell before
# implementation, licenses a route only with its record's evidence, and rejects a
# capability that executes without its proof, refusal and artifact evidence.
#
#   bash scripts/gate_promotion.sh              # static contract, then execute cited evidence
#   bash scripts/gate_promotion.sh --self-test  # broken registries must fail
#   bash scripts/gate_promotion.sh --mutation-check  # manual: every rule has a live case
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REGISTRY=parity/promotion_2_2.toml

# The self-test (scripts/promotion_selftest.py) builds ONE synthetic record
# (2.2A.XT.self_test_cell) with its own owning registries and source files in a temp
# dir, so it never depends on the state of any committed record. The baseline
# synthetic registry must pass the checker; each corruption case mutates only
# synthetic files and must fail with its specific message and rule id, and every
# rule of the checker must have a tagged case. Positive cases must pass. One sanity
# case checks the committed registry passes. Committed files are never written.
#
#   bash scripts/gate_promotion.sh --mutation-check   # manual: switch each rule off in turn;
#                                                     # its cases must stop failing
self_test() {
  python3 scripts/promotion_selftest.py "$@"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi
if [[ "${1:-}" == "--mutation-check" ]]; then
  self_test --mutation-check
  exit $?
fi

group=all
if [[ "${1:-}" == "--evidence-group" && $# -eq 2 ]]; then
  group="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--self-test|--mutation-check|--evidence-group GROUP]" >&2
  exit 2
fi

echo "== 2.2 promotion records =="
EVIDENCE="$(mktemp)"
SELECTED="$(mktemp)"
EMITTED="$EVIDENCE"
trap 'rm -f "$EMITTED" "$SELECTED"' EXIT
if [[ "$group" == all || "$group" == python ]]; then
  python3 scripts/check_promotion_records.py --emit-evidence "$EVIDENCE"
else
  # The required Python group performs full Cargo/Pytest collection checks.
  # Rust groups independently validate the registry and execute their exact
  # cited assertions without paying for the native Python extension.
  PROMOTION_STATIC_ONLY=1 python3 scripts/check_promotion_records.py --emit-evidence "$EVIDENCE"
fi
if [[ "$group" != all ]]; then
  python3 scripts/partition_promotion_evidence.py "$EVIDENCE" "$SELECTED" "$group"
  EVIDENCE="$SELECTED"
fi

# Cited fixtures and closed-route refusal tests are executed, not just resolved.
# Frozen records cite nothing yet, so there may be nothing to run.
if [[ -s "$EVIDENCE" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "FAIL: uv is required; unexecuted Python rows are not promotion evidence"
    exit 1
  fi
  echo "== promotion evidence: executing cited fixtures and refusals =="
  python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_promotion
else
  echo "no promotion fixture cites evidence yet; nothing to execute"
fi
echo "gate_promotion: ok"
