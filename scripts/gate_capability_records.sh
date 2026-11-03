#!/usr/bin/env bash
# Capability gate: versioned registries freeze each release cell before
# implementation, supports a route only with its record's evidence, and rejects a
# capability that executes without its proof, refusal and artifact evidence.
#
#   bash scripts/gate_capability_records.sh              # static contract, then execute cited evidence
#   bash scripts/gate_capability_records.sh --self-test  # broken registries must fail
#   bash scripts/gate_capability_records.sh --mutation-check  # manual: every rule has a live case
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# The self-test (scripts/capability_selftest.py) builds ONE synthetic record
# (xt.cell) with its own owning registries and source files in a temp
# dir, so it never depends on the state of any committed record. The baseline
# synthetic registry must pass the checker; each corruption case mutates only
# synthetic files and must fail with its specific message and rule id, and every
# rule of the checker must have a tagged case. Positive cases must pass. One sanity
# case checks the committed registry passes. Committed files are never written.
#
#   bash scripts/gate_capability_records.sh --mutation-check   # manual: switch each rule off in turn;
#                                                     # its cases must stop failing
self_test() {
  python3 scripts/capability_selftest.py "$@"
}

if [[ "${1:-}" == "--self-test" ]]; then
  python3 scripts/check_prerequisites.py --self-test
  python3 scripts/check_evidence_ledger.py --self-test
  self_test
  exit $?
fi
if [[ "${1:-}" == "--mutation-check" ]]; then
  self_test --mutation-check
  exit $?
fi


# Ordinary capability checks must use the committed registries and fresh evidence.
# Synthetic overrides remain available through the explicit self-test entry points.
for override in CAPABILITY_TRANSPORT_STAGES CAPABILITY_SUPPORT_CELLS \
  CAPABILITY_SUPPORT_REFUSAL_REASONS CAPABILITY_EXTRA_SOURCES CAPABILITY_DISABLE_RULES \
  CAPABILITY_SYNTHETIC_EVIDENCE CAPABILITY_EVIDENCE_CACHE CAPABILITY_STATIC_ONLY; do
  if [[ -n "${!override:-}" ]]; then
    echo "capability gate: $override is not allowed for ordinary checks" >&2
    exit 2
  fi
done

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
python3 scripts/check_prerequisites.py
python3 scripts/check_evidence_ledger.py
for registry in parity/capability_records.toml; do
  echo "== capability records =="
  : > "$EVIDENCE"
  if [[ "$group" == all ]]; then
    python3 scripts/check_capability_records.py "$registry" --emit-evidence "$EVIDENCE"
  else
    # Shards validate every registry, then execute only their assigned rows.
    python3 scripts/check_capability_records.py "$registry" --static-evidence --emit-evidence "$EVIDENCE"
  fi
  if [[ ! -s "$EVIDENCE" ]]; then
    echo "no capability fixture cites evidence yet; nothing to execute"
    continue
  fi
  selected="$EVIDENCE"
  if [[ "$group" != all ]]; then
    python3 scripts/partition_capability_evidence.py "$EVIDENCE" "$SELECTED" "$group"
    selected="$SELECTED"
  fi
  if [[ ! -s "$selected" ]]; then
    echo "no capability evidence in $group shard"
    continue
  fi
  if [[ "$group" == all || "$group" == python ]] && ! command -v uv >/dev/null 2>&1; then
    echo "FAIL: uv is required; unexecuted Python rows are not capability evidence"
    exit 1
  fi
  echo "== capability evidence: executing cited fixtures and refusals =="
  python3 scripts/run_evidence_rows.py "$ROOT" "$selected" "$ROOT" fixture_evidence gate_capability_records
done
echo "gate_capability_records: ok"
