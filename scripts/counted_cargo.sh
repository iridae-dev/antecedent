#!/usr/bin/env bash
# `cargo <args>` that fails when it ran no tests.
#
# libtest exits 0 on "running 0 tests", so a renamed test or module turns a
# filtered `cargo test` line into a no-op that stays green forever. Every gate
# line that filters or names a target runs through here instead:
#
#   bash scripts/counted_cargo.sh test -p antecedent-io --lib posterior
#
# Every `test result:` line is summed; the invocation must have passed at least
# one test and failed none. `MIN_PASSED=N` raises the floor.
set -uo pipefail

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

cargo "$@" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

passed="$(awk '/^test result: ok\./ { n += $4 } END { print n + 0 }' "$log")"
if [[ "$passed" -lt "${MIN_PASSED:-1}" ]]; then
  echo "FAIL: 'cargo $*' passed ${passed} test(s), need at least ${MIN_PASSED:-1}" >&2
  echo "  a renamed test or module makes a filtered gate line a silent no-op" >&2
  exit 1
fi
