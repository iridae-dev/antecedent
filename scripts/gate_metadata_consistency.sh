#!/usr/bin/env bash
# Cross-file metadata consistency gate.
#
# Every check here exists because a fact stated in two places drifted apart and
# nothing caught it. The rule is: one canonical machine-readable source, and
# every restatement validated against it — never two hand-maintained copies.
#
# Run directly, or via scripts/gate_release.sh (which CI's `gates` job runs on
# every PR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --self-test: each broken input, applied alone to an overlay of the repo, must
# fail this gate with the expected message (scripts/selftest_cases.py).
if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" metadata
fi

python3 "$ROOT/scripts/gate_metadata_consistency.py"
