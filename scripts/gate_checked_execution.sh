#!/usr/bin/env bash
# Every licensed estimator must execute from a retained checked operation after
# its builder is gone. The Python gate rejects a cell with no citation, a
# citation that does not run the retained program, and a citation that builds
# a public Study again before that run. It also rejects the linear/Gaussian
# route when that route grows a second compiler or re-enters the public
# builder. Then resolve every remaining citation to an exact non-ignored
# libtest identity, run each distinct assertion once, and attribute failures
# to every licensing cell that cites it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

shard=all
if [[ "${1:-}" == --shard && $# -eq 2 && "$2" =~ ^[0-3]/4$ ]]; then
  shard="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--shard 0/4|1/4|2/4|3/4]" >&2
  exit 2
fi

EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 "$ROOT/scripts/gate_checked_execution.py" "$EVIDENCE" "$shard"

python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_checked_execution
