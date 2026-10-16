#!/usr/bin/env bash
# PAG / LPCMCI gate: inventory honesty + Python smoke. Rust fixtures run in the Rust job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/gate_parity_schema.sh

python3 "$ROOT/scripts/gate_pag.py"

# Rust suites and the Criterion smoke run in the Rust job and gate_release.sh.
echo "== Python conditioning-set provenance facade smoke =="
python_smoke tests/test_discovery_provenance.py

echo "PAG gate PASSED"
