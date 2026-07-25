#!/usr/bin/env bash
# Local Python lint / type gate (ruff + mypy).
# Not part of the wheel-matrix CI job — run before merging Python/PyO3 changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/python"

if [[ ! -d .venv ]]; then
  echo "python/.venv missing; run: cd python && uv venv && uv sync --group dev && maturin develop" >&2
  exit 1
fi

# Prefer uv-run so the gate works without activating the venv.
export CONDA_PREFIX="${CONDA_PREFIX:-}"
# shellcheck disable=SC1091
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # Avoid uv/maturin complaining when both VIRTUAL_ENV and CONDA_PREFIX are set.
  unset CONDA_PREFIX || true
fi

echo "==> ruff check"
uv run ruff check antecedent tests examples

echo "==> ruff format --check"
uv run ruff format --check antecedent tests examples

echo "==> mypy (package + stubs)"
uv run mypy

echo "Python lint/type gate OK"
