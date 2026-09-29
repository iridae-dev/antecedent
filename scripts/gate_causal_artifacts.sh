#!/usr/bin/env bash
# Focused format-0.5 Rust/Python causal query/result artifact conformance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

# The Rust job already runs the antecedent-io artifact tests.
python_smoke tests/test_causal_artifacts.py

echo "gate_causal_artifacts: ok"
