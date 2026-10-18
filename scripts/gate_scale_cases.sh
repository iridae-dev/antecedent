#!/usr/bin/env bash
# Release-only scale cases: the bound `#[ignore]` acceptance tests that are not calibration
# designs (100,000-row fits, 14-million-state enumerations). Each runs in release mode under
# its case's frozen runtime and peak-memory budget; scripts/gate_release_candidate.sh runs
# this gate, and scripts/check_acceptance_manifest.py refuses an ignored bound test unless it
# is wired here.
#
#   bash scripts/gate_scale_cases.sh
#   bash scripts/gate_scale_cases.sh --self-test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--self-test" ]]; then
  python3 scripts/run_scale_cases.py --self-test
  exit 0
fi

python3 scripts/run_scale_cases.py
