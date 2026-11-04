#!/usr/bin/env bash
# Composition exit gate: complete only when seven end-to-end user stories pass from clean
# preparation through independent artifact consumption.
#
#   bash scripts/gate_composition_exit.sh                         # run everything, print the per-story table
#   bash scripts/gate_composition_exit.sh --self-test             # cheap: the verdict logic
#
# It runs the seven Rust stories (crates/antecedent/tests/composition_exit_gate.rs) and the Python stories
# (python/tests/test_composition_exit_gate.py, against the extension already built into python/.venv; build
# it first with `cd python && uv run maturin develop --release`, or set COMPOSITION_EXIT_BUILD_PYTHON=1),
# and joins them with scripts/composition_exit_report.py.
#
# Status per story: PASS / FAIL. Exit status is nonzero on any FAIL. Calibration is printed afterwards
# as an annotation (scripts/calibration_status.py intervals) and never changes the verdict.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    --self-test)
      python3 scripts/composition_exit_report.py --self-test
      echo "gate_composition_exit self-test: ok"
      exit 0
      ;;
    *) echo "usage: gate_composition_exit.sh [--self-test]" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required; unexecuted Python stories are not exit evidence"
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "== composition exit: seven Rust stories =="
cargo test -p antecedent --test composition_exit_gate 2>&1 | tee "$tmp/rust.log" || true

echo "== composition exit: seven Python stories =="
if [[ "${COMPOSITION_EXIT_BUILD_PYTHON:-0}" == "1" ]]; then
  (cd python && uv run maturin develop --release)
fi
(cd python && uv run pytest -v -p no:cacheprovider tests/test_composition_exit_gate.py 2>&1) | tee "$tmp/python.log" || true

echo
status=0
python3 scripts/composition_exit_report.py --rust "$tmp/rust.log" --python "$tmp/python.log" || status=$?

echo
echo "== calibration annotation (informational) =="
python3 scripts/calibration_status.py intervals || true
exit "$status"
