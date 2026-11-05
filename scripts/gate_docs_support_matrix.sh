#!/usr/bin/env bash
# Docs vs support-matrix: generated blocks equal their generators' output.
#
# Run standalone or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --self-test: each broken input, applied alone to an overlay of the repo,
# must fail this gate with the expected message.
if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" docs
fi

python3 "$ROOT/scripts/gate_docs_support_matrix.py"

python3 scripts/short_series_thresholds.py check
