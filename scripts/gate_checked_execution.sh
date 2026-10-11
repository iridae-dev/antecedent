#!/usr/bin/env bash
# Every licensed estimator must execute from a retained checked operation after
# its builder is gone. Resolve every citation to an exact non-ignored libtest
# identity, run each distinct assertion once, and attribute failures to every
# licensing cell that cites it. The support-matrix gate validates the citation
# schema and estimator coverage separately.
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
