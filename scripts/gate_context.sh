#!/usr/bin/env bash
# Context / regime / effects gate: inventory honesty + Python smoke. Rust fixtures run in the Rust job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/gate_parity_schema.sh

python3 "$ROOT/scripts/gate_context.py"

# Rust suites and the Criterion smoke run in the Rust job and gate_release.sh.
echo "== Python EventFrame / panel pooled discovery facade smoke =="
python_smoke tests/test_eventframe_discovery.py tests/test_panel_pooled_discovery.py

echo "Context gate PASSED"
