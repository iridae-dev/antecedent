#!/usr/bin/env bash
# Support-matrix honesty: axes match the live public surface; n/a and licensed
# rows are well-formed; unspecified cells do not exist (default is refused).
#
# Every licensed row's evidence_test / evidence_assertion must be an executing
# test (resolved through `cargo test -- --list`, not ignored; or a collected
# pytest), and, read with the helpers it calls (scripts/test_evidence.py), must
# consume the row's known_truth_fixture when the row claims known truth and build
# every axis value of the row.
#
# Every licensed row's checked_execution names, per licensed estimator, the
# executing test that drops its builder, executes the retained checked plan and
# inspects it (scripts/test_evidence.py checked_execution_problems); the cited
# tests run in scripts/gate_checked_execution.sh.
#
# Run standalone or via scripts/gate_release.sh.
#   bash scripts/gate_support_matrix.sh --self-test   # broken evidence must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/test_evidence_selftest.py"
fi
mode="${1:-all}"
matrix_shard=all
if [[ "$mode" =~ ^--matrix-shard-([0-7])$ ]]; then
  matrix_shard="${BASH_REMATCH[1]}"
  mode=--matrix-only
fi
case "$mode" in all|--core-only|--graphless-only|--matrix-only|--transport-only) ;; *)
  echo "usage: $0 [--self-test|--core-only|--graphless-only|--matrix-only|--matrix-shard-0..7|--transport-only]" >&2; exit 2 ;;
esac

if [[ "$mode" == --transport-only ]]; then
  echo "== transport stage contracts =="
  python3 "$ROOT/scripts/check_transport_stages.py"
  python3 "$ROOT/scripts/generate_calibration_backlog.py" --check
  exit
fi

if [[ "$mode" != --matrix-only ]]; then
  bash scripts/gate_graphless_support.sh
fi
if [[ "$mode" == --graphless-only ]]; then
  exit
fi

python3 "$ROOT/scripts/gate_support_matrix.py" "$matrix_shard"

if [[ "$mode" == --core-only || "$mode" == --matrix-only ]]; then
  exit
fi
echo "== transport stage contracts =="
python3 "$ROOT/scripts/check_transport_stages.py"
python3 "$ROOT/scripts/generate_calibration_backlog.py" --check
