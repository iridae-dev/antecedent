#!/usr/bin/env bash
# 2.2 A exit gate: milestone A is complete only when six end-to-end user stories pass from clean
# preparation through independent artifact consumption, with zero newly introduced unmeasured
# interval coordinates (TODO.md "2.2 A exit gate").
#
#   bash scripts/gate_a_exit.sh                         # run everything, print the per-story table
#   bash scripts/gate_a_exit.sh --require-calibrated    # PENDING_CALIBRATION is a failure (release)
#   bash scripts/gate_a_exit.sh --self-test             # cheap: the checker and the verdict logic
#
# It runs the six Rust stories (crates/antecedent/tests/a_exit_gate.rs), the Python stories
# (python/tests/test_a_exit_gate.py, against the extension already built into python/.venv; build
# it first with `cd python && uv run maturin develop --release`, or set A_EXIT_BUILD_PYTHON=1),
# then scripts/check_a_intervals.py --run-refusals (which also executes the cited closed-route
# refusal tests), and joins them with scripts/a_exit_report.py.
#
# Status per story: PASS / FAIL / PENDING_CALIBRATION. Story 3's "calibrated" and X1's interval
# records are PENDING_CALIBRATION while the coverage records allocated to them in
# parity/promotion_2_2.toml are absent from parity/coverage_records.toml, and PASS automatically
# once they exist and the calibration attestation (scripts/calibration_facets.py) holds. Exit
# status is nonzero on any FAIL; PENDING_CALIBRATION exits 0 unless --require-calibrated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

require_calibrated=()
for arg in "$@"; do
  case "$arg" in
    --self-test)
      python3 scripts/check_a_intervals.py --self-test
      python3 scripts/a_exit_report.py --self-test
      echo "gate_a_exit self-test: ok"
      exit 0
      ;;
    --require-calibrated) require_calibrated=(--require-calibrated) ;;
    *) echo "usage: gate_a_exit.sh [--require-calibrated|--self-test]" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required; unexecuted Python stories are not exit evidence"
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "== A exit gate: six Rust stories =="
cargo test -p antecedent --test a_exit_gate 2>&1 | tee "$tmp/rust.log" || true

echo "== A exit gate: six Python stories =="
if [[ "${A_EXIT_BUILD_PYTHON:-0}" == "1" ]]; then
  (cd python && uv run maturin develop --release)
fi
(cd python && uv run pytest -v -p no:cacheprovider tests/test_a_exit_gate.py 2>&1) | tee "$tmp/python.log" || true

echo "== A exit gate: interval coordinates and calibration state =="
python3 scripts/check_a_intervals.py --run-refusals --json >"$tmp/intervals.json" || true
if [[ ! -s "$tmp/intervals.json" ]]; then
  echo '{"intervals":"FAIL","calibration":{},"errors":["check_a_intervals.py produced no result"]}' >"$tmp/intervals.json"
fi

echo
status=0
python3 scripts/a_exit_report.py --rust "$tmp/rust.log" --python "$tmp/python.log" \
  --intervals "$tmp/intervals.json" ${require_calibrated[@]+"${require_calibrated[@]}"} || status=$?
exit "$status"
