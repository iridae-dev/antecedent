#!/usr/bin/env bash
# Promotion gate: versioned registries freeze each release cell before
# implementation, licenses a route only with its record's evidence, and rejects a
# capability that executes without its proof, refusal and artifact evidence.
#
#   bash scripts/gate_promotion.sh              # static contract, then execute cited evidence
#   bash scripts/gate_promotion.sh --self-test  # broken registries must fail
#   bash scripts/gate_promotion.sh --mutation-check  # manual: every rule has a live case
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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

EVIDENCE="$(mktemp)"
SELECTED="$(mktemp)"
trap 'rm -f "$EVIDENCE" "$SELECTED"' EXIT
python3 scripts/check_2_3_prerequisites.py
for release in 2_2 2_3; do
  registry="parity/promotion_${release}.toml"
  echo "== ${release/_/.} promotion records =="
  : > "$EVIDENCE"
  if [[ "$group" == all ]]; then
    python3 scripts/check_promotion_records.py "$registry" --emit-evidence "$EVIDENCE"
  else
    # Shards validate every registry, then execute only their assigned rows.
    PROMOTION_STATIC_ONLY=1 python3 scripts/check_promotion_records.py "$registry" --emit-evidence "$EVIDENCE"
  fi
  if [[ ! -s "$EVIDENCE" ]]; then
    echo "no promotion fixture cites evidence yet; nothing to execute"
    continue
  fi
  selected="$EVIDENCE"
  if [[ "$group" != all ]]; then
    python3 scripts/partition_promotion_evidence.py "$EVIDENCE" "$SELECTED" "$group"
    selected="$SELECTED"
  fi
  if [[ ! -s "$selected" ]]; then
    echo "no ${release/_/.} promotion evidence in $group shard"
    continue
  fi
  if [[ "$group" == all || "$group" == python ]] && ! command -v uv >/dev/null 2>&1; then
    echo "FAIL: uv is required; unexecuted Python rows are not promotion evidence"
    exit 1
  fi
  echo "== ${release/_/.} promotion evidence: executing cited fixtures and refusals =="
  python3 scripts/run_evidence_rows.py "$ROOT" "$selected" "$ROOT" fixture_evidence gate_promotion
done
echo "gate_promotion: ok"
