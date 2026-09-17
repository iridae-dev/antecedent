#!/usr/bin/env bash
# Coverage citation gate.
#
# A licensed row that quotes a measured coverage cites the test that produced
# it, in the form `0.895 (v19_static_calibration::path_specific_two_path_*)` or
# `(crates/antecedent/tests/v19_derivative_calibration.rs::ade_..._coverage)`.
# The 2026-09-15 traceability pass found quoted numbers that no test had
# produced on the current tree; a citation that names a test which does not
# exist is the cheapest such drift to catch mechanically. This gate resolves
# every `<module>::<fn>` citation in parity/support_licensed.toml against the
# test functions actually defined in crates/antecedent/tests/<module>.rs
# (`fn name` and the `name => (...)` entries of the calibration macros),
# expanding `{a,b}` alternatives and `*` globs, and fails on any citation that
# matches no test. It extends scripts/gate_evidence_reachability.sh, which
# checks the evidence_test / evidence_assertion columns but not the prose.
#
# It then requires every coverage figure in the prose to be attributed
# (scripts/coverage_citations.py): a figure either cites the coverage record whose
# observed value it states, or says it is not a registry value (a probe or an
# earlier measurement the registry does not carry). Known-truth values, standard
# errors and pinned values are not coverage figures and are never rejected.
#
# Run directly, or via scripts/gate_release.sh (CI's `gates` job, every PR).
# --self-test: broken citations must fail this gate (scripts/selftest_cases.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" citations
fi

python3 - <<'PY'
import fnmatch
import itertools
import re
import sys
import tomllib
from pathlib import Path

root = Path(".")
tests_dir = root / "crates/antecedent/tests"
lic = tomllib.load(open(root / "parity/support_licensed.toml", "rb"))

# ------------------------------------------------ test functions per module
FN_RE = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]", re.M)
# Calibration macros (`dag_coverage! { name => (cell, regime); ... }`) generate one
# test per `name =>` entry.
MACRO_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*=>", re.M)


def test_fns(module: str):
    path = tests_dir / f"{module}.rs"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    return set(FN_RE.findall(text)) | set(MACRO_RE.findall(text))


fns_cache = {}

# ------------------------------------------------ citations in the prose
# `v19_static_calibration::name`, `crates/antecedent/tests/v19_x.rs::name`, and the
# shorthand `::name` that continues the last-named module inside one citation.
CITE_RE = re.compile(
    r"(?:crates/antecedent/tests/)?(v1(?:9|10)_[a-z0-9_]+)(?:\.rs)?::([A-Za-z0-9_{},*]+)"
    r"((?:,\s*::[A-Za-z0-9_{},*]+)*)"
)
CONT_RE = re.compile(r"::([A-Za-z0-9_{},*]+)")


def expand(pattern: str):
    """`a_{x,y}_*` -> the glob patterns `a_x_*`, `a_y_*`."""
    parts = re.split(r"(\{[^}]*\})", pattern)
    alts = [p[1:-1].split(",") if p.startswith("{") else [p] for p in parts]
    return ["".join(combo) for combo in itertools.product(*alts)]


fail = []
checked = 0
for cell in lic.get("cell", []):
    row = f"{cell['query']}/{cell['graph_class']}/{cell['structure']}/{cell['inference']}/{cell['validation']}"
    lim = str(cell.get("limitations", ""))
    for m in CITE_RE.finditer(lim):
        module = m.group(1)
        names = [m.group(2)] + CONT_RE.findall(m.group(3) or "")
        if module not in fns_cache:
            fns_cache[module] = test_fns(module)
        fns = fns_cache[module]
        if fns is None:
            fail.append(f"{row}: cites {module}::{names[0]} but crates/antecedent/tests/{module}.rs does not exist")
            continue
        for name in names:
            checked += 1
            pats = expand(name.rstrip(".,;"))
            if not any(fnmatch.fnmatchcase(fn, pat) for pat in pats for fn in fns):
                fail.append(f"{row}: cites {module}::{name} but no such test fn in crates/antecedent/tests/{module}.rs")

if fail:
    print("Coverage citation gate FAILED:")
    for f in sorted(set(fail)):
        print(" -", f)
    sys.exit(1)

print(f"Coverage citations OK ({checked} citations across {len(fns_cache)} test modules resolve to existing test fns)")
PY

python3 scripts/coverage_citations.py check
