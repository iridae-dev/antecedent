#!/usr/bin/env bash
# Evidence reachability gate.
#
# The 2026-08-18 audit found that every mechanized check in this repository
# held and every check that depended on an author remembering had drifted.
# This gate mechanizes the two that drifted worst:
#
#   1. Cited evidence must be executed evidence. Three pinned-baseline
#      general-ID fixtures — and, found by this gate's own dry run,
#      uncertainty_routing and temporal_pressure_defect — sat on disk loaded by
#      no test while provenance, the oracle ledger, and generated docs
#      described the comparison in the present tense.
#
#   2. A citation must not imply theorem inheritance the code has not earned.
#      estimate.aipw cited the DML paper for two releases without recording
#      that the implementation has no cross-fitting; there was simply no field
#      to put the deviation in. Now there is, and new records cannot merge
#      without it.
#
# Run directly, or via scripts/gate_release.sh (CI's `gates` job, every PR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --self-test: each broken input, applied alone to an overlay of the repo, must
# fail this gate with the expected message (scripts/selftest_cases.py).
if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" reachability
fi

python3 "$ROOT/scripts/gate_evidence_reachability.py"
