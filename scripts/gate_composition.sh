#!/usr/bin/env bash
# 1.10 composition gate: ledger rows in parity/compiler.toml drive invocations.
#
# Every row must name an executing test (`evidence_test` + `evidence_assertion`).
# The gate proves each one ran:
#   - the assertion is resolved to its full test name (`cargo test -- --list`;
#     lib tests are module-qualified, e.g. `execution::tests::<fn>`) and run
#     with `--exact`; exactly one test must pass and none may fail;
#   - `composition_filter` is a cargo substring filter, not a regex, so a
#     `a|b` value (or a list) is split and each part runs on its own; every
#     part must pass at least one test and fail none;
#   - Python rows run the pytest node; at least one test passes, none fail.
# Counts are parsed from cargo's `test result: ... N passed; M failed` lines and
# pytest's summary line, never from a loose grep that `0 passed` satisfies.
#
#   bash scripts/gate_composition.sh              # full gate
#   bash scripts/gate_composition.sh --self-test  # broken inputs must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# run_rows ROOT LEDGER — execute every ledger row; non-zero exit on any failure.
run_rows() {
  python3 "$ROOT/scripts/run_evidence_rows.py" "$1" "$2" "$ROOT"
}

self_test() {
  local tmp status out
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  status=0
  # A throwaway workspace with passing and panicking lib / integration tests.
  mkdir -p "$tmp/crates/selftest/src" "$tmp/crates/selftest/tests" "$tmp/parity" "$tmp/python/tests"
  cat >"$tmp/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = ["crates/selftest"]
TOML
  cat >"$tmp/crates/selftest/Cargo.toml" <<'TOML'
[package]
name = "selftest"
version = "0.0.0"
edition = "2021"
publish = false
TOML
  cat >"$tmp/crates/selftest/src/lib.rs" <<'RS'
pub mod inner;
#[cfg(test)]
mod tests {
    #[test]
    fn lib_passes() {}
    #[test]
    fn lib_panics() {
        panic!("deliberately broken");
    }
}
RS
  cat >"$tmp/crates/selftest/src/inner.rs" <<'RS'
#[cfg(test)]
mod tests {
    #[test]
    fn lib_passes() {}
}
RS
  cat >"$tmp/crates/selftest/tests/it.rs" <<'RS'
#[test]
fn it_passes() {}
#[test]
fn it_panics() {
    panic!("deliberately broken");
}
RS
  cat >"$tmp/python/tests/test_selftest.py" <<'PYT'
def test_passes():
    pass


def test_fails():
    raise AssertionError("deliberately broken")
PYT
  row() { # id test assertion [filter]
    printf '[[capabilities]]\nid = "%s"\nevidence_test = "%s"\nevidence_assertion = "%s"\n' "$1" "$2" "$3"
    if [[ -n "${4:-}" ]]; then printf 'composition_filter = "%s"\n' "$4"; fi
    printf '\n'
  }
  export CARGO_TARGET_DIR="$tmp/target"
  # Positive controls: must pass, including a module-qualified lib test.
  {
    row good.lib crates/selftest/src/lib.rs lib_passes
    row good.inner crates/selftest/src/inner.rs lib_passes
    row good.it crates/selftest/tests/it.rs it_passes it_passes
    row good.py python/tests/test_selftest.py test_passes
  } >"$tmp/parity/good.toml"
  if ! out="$(run_rows "$tmp" parity/good.toml 2>&1)"; then
    echo "SELF-TEST FAIL: passing rows were rejected"; echo "$out"; status=1
  else
    echo "self-test ok: passing controls accepted"
  fi
  # Each broken row, alone, must fail the gate.
  local -a cases=(
    "panic.lib|crates/selftest/src/lib.rs|lib_panics|"
    "panic.it|crates/selftest/tests/it.rs|it_panics|"
    "panic.filter|crates/selftest/tests/it.rs|it_passes|it_"
    "missing.assertion|crates/selftest/tests/it.rs|no_such_test|"
    "pipe.filter|crates/selftest/tests/it.rs|it_passes|it_passes|zz_no_match"
    "fail.py|python/tests/test_selftest.py|test_fails|"
    "missing.py|python/tests/test_selftest.py|test_absent|"
    "no.evidence|||"
  )
  local spec cid test assertion filt
  for spec in "${cases[@]}"; do
    # The last field keeps any further `|`, so pipe.filter's filter stays `a|b`.
    IFS='|' read -r cid test assertion filt <<<"$spec"
    if [[ "$cid" == "no.evidence" ]]; then
      printf '[[capabilities]]\nid = "no.evidence"\n' >"$tmp/parity/case.toml"
    else
      row "$cid" "$test" "$assertion" "$filt" >"$tmp/parity/case.toml"
    fi
    if out="$(run_rows "$tmp" parity/case.toml 2>&1)"; then
      echo "SELF-TEST FAIL: broken row '$cid' passed the gate"; echo "$out"; status=1
    else
      echo "self-test ok: '$cid' fails: $(grep -m1 '^FAIL' <<<"$out")"
    fi
  done
  # An empty ledger fails.
  : >"$tmp/parity/empty.toml"
  if run_rows "$tmp" parity/empty.toml >/dev/null 2>&1; then
    echo "SELF-TEST FAIL: an empty ledger passed"; status=1
  else
    echo "self-test ok: empty ledger fails"
  fi
  unset CARGO_TARGET_DIR
  if [[ "$status" -ne 0 ]]; then
    return 1
  fi
  echo "gate_composition self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if [[ "${SKIP_PYTHON_SMOKE:-0}" == "1" ]]; then
  echo "FAIL: SKIP_PYTHON_SMOKE=1 is not composition evidence"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required for the composition Python smoke"
  exit 1
fi

REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "== existing evidence gates over composition records =="
bash scripts/gate_parity_schema.sh
bash scripts/gate_provenance_schema.sh
bash scripts/gate_metadata_consistency.sh
bash scripts/gate_evidence_reachability.sh

echo "== 1.10 composition consuming tests =="
run_rows "$ROOT" parity/compiler.toml

echo "revision: ${REVISION}"
echo "gate_composition: ok"
