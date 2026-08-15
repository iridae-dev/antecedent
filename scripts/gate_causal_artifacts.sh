#!/usr/bin/env bash
# Focused format-0.3 Rust/Python causal query/result artifact conformance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo test -p antecedent-io causal_artifact --no-fail-fast

if [[ "${SKIP_PYTHON_SMOKE:-0}" == "1" ]]; then
  echo "SKIP_PYTHON_SMOKE=1; skipping (covered by python-wheels CI)"
elif ! command -v uv >/dev/null 2>&1; then
  echo "WARN: uv not on PATH; skipping Python facade smoke (covered by python-wheels CI)"
else
  (cd python && uv run pytest -q tests/test_causal_artifacts.py)
fi

echo "gate_causal_artifacts: ok"
