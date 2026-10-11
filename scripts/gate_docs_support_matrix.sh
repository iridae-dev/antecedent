#!/usr/bin/env bash
# Docs vs support-matrix honesty: capabilities/comparison must point at the
# matrix and must not treat closed/n-a cells as licensed public analyze support.
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
