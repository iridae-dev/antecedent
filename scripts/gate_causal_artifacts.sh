#!/usr/bin/env bash
# Focused format-0.5 Rust/Python causal query/result artifact conformance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/counted_cargo.sh test -p antecedent-io causal_artifact --no-fail-fast

python_smoke tests/test_causal_artifacts.py

echo "gate_causal_artifacts: ok"
