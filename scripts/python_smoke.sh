#!/usr/bin/env bash
# Shared Python facade smoke for the feature gates. Source it, then call from
# the repo root:
#
#   source scripts/python_smoke.sh
#   python_smoke tests/test_a.py tests/test_b.py
#
# A missing `uv` or SKIP_PYTHON_SMOKE=1 fails the gate, the same rule
# gate_composition.sh applies: a smoke that can be skipped by an environment
# variable or a missing tool is not evidence. A developer without uv may set
# ALLOW_SKIP_PYTHON_SMOKE=1 to skip locally; the release-candidate gate refuses
# both variables. The pytest run itself must pass at least one test (a broken
# native extension turns modules into skips and pytest still exits 0).
python_smoke() {
  local allow="${ALLOW_SKIP_PYTHON_SMOKE:-0}" log status passed
  if [[ "${SKIP_PYTHON_SMOKE:-0}" == "1" ]]; then
    if [[ "$allow" == "1" ]]; then
      echo "SKIP_PYTHON_SMOKE=1 with ALLOW_SKIP_PYTHON_SMOKE=1; skipping (local run only)"
      return 0
    fi
    echo "FAIL: SKIP_PYTHON_SMOKE=1 is not evidence (set ALLOW_SKIP_PYTHON_SMOKE=1 for a local run)" >&2
    return 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    if [[ "$allow" == "1" ]]; then
      echo "WARN: uv not on PATH; skipping Python facade smoke (ALLOW_SKIP_PYTHON_SMOKE=1)"
      return 0
    fi
    echo "FAIL: uv is required for the Python facade smoke (ALLOW_SKIP_PYTHON_SMOKE=1 to skip locally)" >&2
    return 1
  fi
  log="$(mktemp)"
  status=0
  (
    cd python
    unset CONDA_PREFIX || true
    uv run pytest -q -rs -p no:cacheprovider "$@"
  ) >"$log" 2>&1 || status=$?
  cat "$log"
  if [[ "$status" -ne 0 ]]; then
    rm -f "$log"
    return "$status"
  fi
  passed="$(grep -Eo '[0-9]+ passed' "$log" | tail -1 | grep -Eo '[0-9]+' || true)"
  rm -f "$log"
  if [[ -z "$passed" || "$passed" -lt 1 ]]; then
    echo "FAIL: the Python smoke ran no passing test (skipped or empty selection)" >&2
    return 1
  fi
}
