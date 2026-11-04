#!/usr/bin/env bash
# Capability gate: a route is supported only with its record's evidence, and a
# capability that executes without its proof, refusal and artifact evidence is rejected.
#
#   bash scripts/gate_capability_records.sh              # static contract, cited evidence resolved by source reading
#   bash scripts/gate_capability_records.sh --resolve    # the same, with cited tests listed by cargo (builds, runs nothing)
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

resolve=0
if [[ "${1:-}" == "--resolve" && $# -eq 1 ]]; then
  resolve=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--self-test|--mutation-check|--resolve]" >&2
  exit 2
fi

EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 scripts/check_prerequisites.py
python3 scripts/check_evidence_ledger.py
for registry in parity/capability_records.toml; do
  echo "== capability records =="
  : > "$EVIDENCE"
  if [[ "$resolve" -eq 1 ]]; then
    # Release candidate: every cited test is listed by cargo (compiled, non-ignored).
    python3 scripts/check_capability_records.py "$registry" --emit-evidence "$EVIDENCE"
  else
    python3 scripts/check_capability_records.py "$registry" --static-evidence --emit-evidence "$EVIDENCE"
  fi
  if [[ ! -s "$EVIDENCE" ]]; then
    echo "no capability fixture cites evidence yet; nothing to resolve"
    continue
  fi
  # Resolution only: the cited tests run in the engineering lane, and the cited
  # scientific ones through scripts/scientific_records.py run (owed only).
  echo "== capability evidence: cited fixtures and refusals resolve to exact tests =="
  python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_capability_records --static
done
echo "gate_capability_records: ok"
