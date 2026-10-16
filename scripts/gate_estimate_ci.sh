#!/usr/bin/env bash
# Estimate/CI parity gate: inventory honesty + Python smoke. Rust fixtures run in the Rust job.
# black-box / Exact pins diverge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/gate_parity_schema.sh

python3 "$ROOT/scripts/gate_estimate_ci.py"

# The Rust CI job already runs these suites (`cargo test --workspace`).
echo "== Python max_cond_size (PCMCI family) facade smoke =="
python_smoke tests/test_discovery_provenance.py

echo "estimate_ci parity gate: ok"
