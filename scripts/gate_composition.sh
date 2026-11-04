#!/usr/bin/env bash
# Composition gate: ledger rows in parity/compiler.toml are the composition evidence.
#
# Every row must name an executing test (`evidence_test` + `evidence_assertion`).
# This gate resolves each one without running it (scripts/run_evidence_rows.py
# --static): the assertion must be a declared, non-ignored `#[test]` compiled into
# its cargo target (or a Python test that pytest can name). The tests themselves run
# in the engineering lane (`bash scripts/test.sh`), and any row that is also a cited
# scientific assertion runs through `scripts/scientific_records.py run`. `--self-test`
# runs the executing runner against a throwaway workspace with broken rows.
#
#   bash scripts/gate_composition.sh              # full gate
#   bash scripts/gate_composition.sh --self-test  # broken inputs must fail
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# run_rows ROOT LEDGER [--static] — execute (or only resolve) every ledger row; non-zero exit on any failure.
run_rows() {
  python3 "$ROOT/scripts/run_evidence_rows.py" "$1" "$2" "$ROOT" capabilities gate_composition "${@:3}"
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
  row() { # id test assertion [extra toml line]
    printf '[[capabilities]]\nid = "%s"\nevidence_test = "%s"\nevidence_assertion = "%s"\n' "$1" "$2" "$3"
    if [[ -n "${4:-}" ]]; then printf '%s\n' "$4"; fi
    printf '\n'
  }
  export CARGO_TARGET_DIR="$tmp/target"
  # Positive controls: must pass, including a module-qualified lib test.
  {
    row good.lib crates/selftest/src/lib.rs lib_passes
    row good.inner crates/selftest/src/inner.rs lib_passes
    row good.it crates/selftest/tests/it.rs it_passes
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
    "missing.assertion|crates/selftest/tests/it.rs|no_such_test|"
    "filter.unsupported|crates/selftest/tests/it.rs|it_passes|composition_filter = \"it_\""
    "fail.py|python/tests/test_selftest.py|test_fails|"
    "missing.py|python/tests/test_selftest.py|test_absent|"
    "no.evidence|||"
  )
  local spec cid test assertion filt
  for spec in "${cases[@]}"; do
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
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--evidence-only" ) ]]; then
  echo "usage: $0 [--self-test|--evidence-only]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required; unexecuted Python rows are not composition evidence"
  exit 1
fi

REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [[ "${1:-}" != "--evidence-only" ]]; then
  echo "== existing evidence gates over composition records =="
  bash scripts/gate_parity_schema.sh
  bash scripts/gate_provenance_schema.sh
  bash scripts/gate_metadata_consistency.sh
  bash scripts/gate_evidence_reachability.sh
fi

echo "== composition consuming tests resolve (they run in the engineering and scientific lanes) =="
run_rows "$ROOT" parity/compiler.toml --static

echo "revision: ${REVISION}"
echo "gate_composition: ok"
