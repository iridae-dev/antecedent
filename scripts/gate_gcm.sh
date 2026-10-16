#!/usr/bin/env bash
# GCM gate: inventory honesty. Rust fixtures run in the Rust job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/gate_parity_schema.sh

python3 "$ROOT/scripts/gate_gcm.py"

# Rust suites, including the gaussian-process feature, and the Criterion smoke
# run in the Rust job and gate_release.sh.
echo "GCM gate PASSED"
