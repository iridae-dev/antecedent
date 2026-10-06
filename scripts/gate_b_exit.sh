#!/usr/bin/env bash
# 2.2 B exit gate (SCAFFOLD): milestone B is complete only when each B package's end-to-end user
# story passes from clean preparation through independent artifact consumption, and every 2.2
# record (A and B) ships zero unmeasured interval coordinates (TODO.md B8).
#
#   bash scripts/gate_b_exit.sh                          # run registered stories, print the table
#   bash scripts/gate_b_exit.sh --require-calibrated     # PENDING_CALIBRATION is a failure
#   bash scripts/gate_b_exit.sh --require-implemented    # PENDING_IMPLEMENTATION is a failure
#   bash scripts/gate_b_exit.sh --release                # both (the 2.2 cut)
#   bash scripts/gate_b_exit.sh --self-test              # cheap: the verdict logic and the checker
#
# REGISTRY: each B package adds its end-to-end story test files to `B_PACKAGES` in
# scripts/b_exit_report.py (the single registry; `rust` = (crate, integration-test target) pairs,
# `python` = test files under python/). B1-B6 each register one story in
# crates/antecedent/tests/b_exit_gate.rs. Do not register a test that does not exist.
# A package can read PASS only with registered story files AND an `evidence` list (route-evidence
# test names that must be defined in a story file or an already-cited record fixture and must appear
# as passed in the story's run log; pytest runs with -rA so passes are listed by name). An allocated
# coverage id (any record of the package, carried_forward included) never passes without a
# calibration reading. See STORY RULES in scripts/b_exit_report.py.
# The stories are run here from the registry, the interval-coordinate check
# (scripts/check_interval_coordinates.py, all 2.2 records) runs with its refusal tests, and
# b_exit_report.py joins them.
#
# Status per package: PASS / FAIL / PENDING_IMPLEMENTATION / PENDING_CALIBRATION (see
# scripts/b_exit_report.py). Exit status is nonzero on any FAIL; the two PENDING states exit 0
# unless their --require flag (or --release) is given. The Python stories run against the extension
# already built into python/.venv (`cd python && uv run maturin develop --release`, or set
# B_EXIT_BUILD_PYTHON=1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

flags=()
for arg in "$@"; do
  case "$arg" in
    --self-test)
      python3 scripts/check_interval_coordinates.py --self-test
      python3 scripts/b_exit_report.py --self-test
      echo "gate_b_exit self-test: ok"
      exit 0
      ;;
    --require-calibrated | --require-implemented | --release) flags+=("$arg") ;;
    *) echo "usage: gate_b_exit.sh [--require-calibrated|--require-implemented|--release|--self-test]" >&2; exit 2 ;;
  esac
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
: >"$tmp/runs.tsv"

tests="$(python3 scripts/b_exit_report.py --list-tests)"
if [[ -n "$tests" ]]; then
  if ! command -v uv >/dev/null 2>&1 && grep -q '^python' <<<"$tests"; then
    echo "FAIL: uv is required; unexecuted Python stories are not exit evidence"
    exit 1
  fi
  if [[ "${B_EXIT_BUILD_PYTHON:-0}" == "1" ]] && grep -q '^python' <<<"$tests"; then
    (cd python && uv run maturin develop --release)
  fi
  n=0
  while IFS=$'\t' read -r kind pkg a b; do
    n=$((n + 1))
    log="$tmp/run$n.log"
    rc=0
    if [[ "$kind" == "rust" ]]; then
      echo "== B exit gate: $pkg rust story $a/$b =="
      cargo test -p "$a" --test "$b" >"$log" 2>&1 || rc=$?
      target="$a/$b"
    else
      echo "== B exit gate: $pkg python story $a =="
      (cd python && uv run pytest -q -rA -p no:cacheprovider "$a") >"$log" 2>&1 || rc=$?
      target="$a"
    fi
    tail -n 5 "$log"
    printf '%s\t%s\t%s\t%s\t%s\n' "$kind" "$pkg" "$target" "$rc" "$log" >>"$tmp/runs.tsv"
  done <<<"$tests"
else
  echo "== B exit gate: no story tests registered yet (scripts/b_exit_report.py B_PACKAGES) =="
fi

echo "== B exit gate: interval coordinates and calibration state (all 2.2 records) =="
python3 scripts/check_interval_coordinates.py --run-refusals --json >"$tmp/intervals.json" || true
if [[ ! -s "$tmp/intervals.json" ]]; then
  echo '{"intervals":"FAIL","calibration":{},"errors":["check_interval_coordinates.py produced no result"]}' >"$tmp/intervals.json"
fi

echo
status=0
python3 scripts/b_exit_report.py --runs "$tmp/runs.tsv" --intervals "$tmp/intervals.json" \
  ${flags[@]+"${flags[@]}"} || status=$?
exit "$status"
