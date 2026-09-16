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
  python3 - "$1" "$2" "$ROOT" <<'PY'
import re
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1]).resolve()
ledger = Path(sys.argv[2])
repo = Path(sys.argv[3]).resolve()

CARGO_RESULT = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;", re.M)
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?)\b")


def cargo_counts(log: str) -> tuple[int, int]:
    """(passed, failed) summed over every test binary cargo ran."""
    results = CARGO_RESULT.findall(log)
    passed = sum(int(p) for _, p, _ in results)
    failed = max(sum(int(f) for _, _, f in results), sum(s == "FAILED" for s, _, _ in results))
    return passed, failed


def pytest_counts(log: str) -> tuple[int, int]:
    """(passed, failed+errors) from pytest's final summary line."""
    summary = [ln for ln in log.splitlines() if PYTEST_COUNT.search(ln) and " in " in ln]
    if not summary:
        return 0, 0
    counts = {"passed": 0, "failed": 0}
    for n, kind in PYTEST_COUNT.findall(summary[-1]):
        key = "passed" if kind == "passed" else "failed"
        counts[key] += int(n)
    return counts["passed"], counts["failed"]


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


failures: list[str] = []


def check(label: str, cmd: list[str], cwd: Path, counter, exact: bool) -> None:
    code, log = run(cmd, cwd)
    passed, failed = counter(log)
    wanted = "exactly 1" if exact else "at least 1"
    ok = code == 0 and failed == 0 and (passed == 1 if exact else passed >= 1)
    if not ok:
        tail = "\n".join(log.splitlines()[-40:])
        failures.append(label)
        print(
            f"FAIL: {label}: exit={code} passed={passed} failed={failed} "
            f"(want {wanted} passed, 0 failed)\n  $ {' '.join(cmd)}\n{tail}"
        )
    else:
        print(f"ok: {label} ({passed} passed)")


def lib_module(test_rel: str) -> str:
    """`crates/x/src/a/b.rs` -> `a::b::`; `src/lib.rs` / `mod.rs` map to their parent."""
    parts = Path(test_rel).parts
    rel = list(parts[parts.index("src") + 1 :])
    rel[-1] = rel[-1].removesuffix(".rs")
    if rel[-1] in ("lib", "mod"):
        rel = rel[:-1]
    return "".join(f"{p}::" for p in rel)


def child_module_dir(test_rel: str) -> Path:
    """Directory holding the file modules declared by `test_rel`."""
    path = root / test_rel
    if path.name in ("lib.rs", "mod.rs"):
        return path.parent
    return path.with_suffix("")


def resolve_rust(crate: str, target: list[str], assertion: str, test_rel: str, module: str | None):
    code, log = run(["cargo", "test", "-p", crate, *target, "--", "--list"], root)
    if code != 0:
        return None, f"`cargo test -p {crate} {' '.join(target)} -- --list` failed:\n{log[-2000:]}"
    names = [ln[: -len(": test")] for ln in log.splitlines() if ln.endswith(": test")]
    hits = [n for n in names if n.rsplit("::", 1)[-1] == assertion]
    if module is not None:
        # Keep tests defined in this file: inside its module, and not inside a
        # child module that lives in its own file.
        children = child_module_dir(test_rel)

        def in_file(name: str) -> bool:
            if not name.startswith(module):
                return False
            first = name[len(module):].split("::", 1)[0]
            return not ((children / f"{first}.rs").is_file() or (children / first / "mod.rs").is_file())

        hits = [n for n in hits if in_file(n)]
    if len(hits) != 1:
        return None, f"assertion {assertion!r} resolves to {hits or 'no test'} in {crate} {' '.join(target)}"
    return hits[0], None


def filters(value) -> list[str]:
    if value is None:
        return []
    parts = value if isinstance(value, list) else str(value).split("|")
    return [p.strip() for p in parts if isinstance(p, str) and p.strip()]


rows = tomllib.loads((root / ledger).read_text()).get("capabilities", [])
if not rows:
    print(f"FAIL: {ledger} has no [[capabilities]] rows")
    sys.exit(1)

for row in rows:
    cid = row.get("id", "<no id>")
    test_rel = row.get("evidence_test")
    assertion = row.get("evidence_assertion")
    if not test_rel or not assertion:
        failures.append(cid)
        print(f"FAIL: {cid}: no evidence_test/evidence_assertion; every composition row must execute")
        continue

    if test_rel.endswith(".py"):
        if not test_rel.startswith("python/"):
            failures.append(cid)
            print(f"FAIL: {cid}: Python evidence {test_rel} is not under python/")
            continue
        node = f"{test_rel.removeprefix('python/')}::{assertion}"
        cmd = ["uv", "run", "--project", str(repo / "python"), "pytest", "-q", "-p", "no:cacheprovider", node]
        check(cid, cmd, root / "python", pytest_counts, exact=False)
        for part in filters(row.get("composition_filter")):
            failures.append(f"{cid}.filter")
            print(f"FAIL: {cid}: composition_filter {part!r} is not supported on a Python row")
        continue

    m = re.fullmatch(r"crates/([^/]+)/(tests|src)/(.+)\.rs", test_rel)
    if not m:
        failures.append(cid)
        print(f"FAIL: {cid}: evidence_test {test_rel} is not a known layout")
        continue
    crate, kind, stem = m.groups()
    if kind == "tests":
        target = ["--test", stem.split("/")[0]]
        module = None
    else:
        target = ["--lib"]
        module = lib_module(test_rel)
    name, problem = resolve_rust(crate, target, assertion, test_rel, module)
    if problem:
        failures.append(cid)
        print(f"FAIL: {cid}: {problem}")
        continue
    check(cid, ["cargo", "test", "-p", crate, *target, "--", "--exact", name], root, cargo_counts, exact=True)
    for part in filters(row.get("composition_filter")):
        check(f"{cid}.filter[{part}]", ["cargo", "test", "-p", crate, *target, "--", part], root, cargo_counts, exact=False)

if failures:
    print(f"gate_composition: {len(failures)} failing row(s): {', '.join(failures)}")
    sys.exit(1)
print(f"gate_composition rows: ok ({len(rows)} rows)")
PY
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
