#!/usr/bin/env bash
# Composition exit gate: complete only when seven end-to-end user stories pass from clean
# preparation through independent artifact consumption, with zero newly introduced unmeasured
# interval coordinates.
#
#   bash scripts/gate_composition_exit.sh                         # run everything, print the per-story table
#   bash scripts/gate_composition_exit.sh --require-calibrated    # PENDING_CALIBRATION is a failure (release)
#   bash scripts/gate_composition_exit.sh --self-test             # cheap: the checker and the verdict logic
#
# It runs the seven Rust stories (crates/antecedent/tests/composition_exit_gate.rs), the Python stories
# (python/tests/test_composition_exit_gate.py, against the extension already built into python/.venv; build
# it first with `cd python && uv run maturin develop --release`, or set COMPOSITION_EXIT_BUILD_PYTHON=1),
# then scripts/check_interval_surfaces.py --run-refusals (which also executes the cited closed-route
# refusal tests), and joins them with scripts/composition_exit_report.py.
#
# Status per story: PASS / FAIL / PENDING_CALIBRATION. Story 3's "calibrated" and the multi-source interval
# records are PENDING_CALIBRATION while the coverage records allocated to them in
# parity/promotion.toml are absent from parity/coverage_records.toml, and PASS automatically
# once they exist and the calibration attestation (scripts/calibration_facets.py) holds. Exit
# status is nonzero on any FAIL; PENDING_CALIBRATION exits 0 unless --require-calibrated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

require_calibrated=()
for arg in "$@"; do
  case "$arg" in
    --self-test)
      python3 scripts/check_interval_surfaces.py --self-test
      python3 scripts/composition_exit_report.py --self-test
      echo "gate_composition_exit self-test: ok"
      exit 0
      ;;
    --require-calibrated) require_calibrated=(--require-calibrated) ;;
    *) echo "usage: gate_composition_exit.sh [--require-calibrated|--self-test]" >&2; exit 2 ;;
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

echo "== composition exit: interval coordinates and calibration state =="
python3 scripts/check_interval_surfaces.py --run-refusals --json >"$tmp/intervals.json" || true
if [[ ! -s "$tmp/intervals.json" ]]; then
  echo '{"intervals":"FAIL","calibration":{},"errors":["check_interval_surfaces.py produced no result"]}' >"$tmp/intervals.json"
fi

echo
status=0
python3 scripts/composition_exit_report.py --rust "$tmp/rust.log" --python "$tmp/python.log" \
  --intervals "$tmp/intervals.json" ${require_calibrated[@]+"${require_calibrated[@]}"} || status=$?
exit "$status"
