#!/usr/bin/env bash
# Focused format-0.3 Rust/Python causal query/result artifact conformance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo test -p antecedent-io causal_artifact --no-fail-fast
(cd python && uv run pytest -q tests/test_causal_artifacts.py)

echo "gate_causal_artifacts: ok"
