#!/usr/bin/env python3
import fnmatch
import itertools
import re
import sys
import tomllib
from pathlib import Path

root = Path(".")
tests_dir = root / "crates/antecedent/tests"
with open(root / "parity/support_licensed.toml", "rb") as handle:
    lic = tomllib.load(handle)

# ------------------------------------------------ test functions per module
FN_RE = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]", re.M)
# Calibration macros (`dag_coverage! { name => (cell, regime); ... }`) generate one
# test per `name =>` entry.
MACRO_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*=>", re.M)


def test_fns(module: str):
    path = tests_dir / f"{module}.rs"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore", encoding="utf-8")
    return set(FN_RE.findall(text)) | set(MACRO_RE.findall(text))


fns_cache = {}

# ------------------------------------------------ citations in the prose
# `static_calibration::name`, `crates/antecedent/tests/derivative_calibration.rs::name`, and the
# shorthand `::name` that continues the last-named module inside one citation.
MODULES = sorted((p.stem for p in tests_dir.glob("*.rs")), key=len, reverse=True)
CITE_RE = re.compile(
    r"(?:crates/antecedent/tests/)?(?<![A-Za-z0-9_])("
    + "|".join(re.escape(m) for m in MODULES)
    + r")(?:\.rs)?::([A-Za-z0-9_{},*]+)"
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
            fail.append(
                f"{row}: cites {module}::{names[0]} but crates/antecedent/tests/{module}.rs does not exist"
            )
            continue
        for name in names:
            checked += 1
            pats = expand(name.rstrip(".,;"))
            if not any(fnmatch.fnmatchcase(fn, pat) for pat in pats for fn in fns):
                fail.append(
                    f"{row}: cites {module}::{name} but no such test fn in crates/antecedent/tests/{module}.rs"
                )

if fail:
    print("Coverage citation gate FAILED:")
    for f in sorted(set(fail)):
        print(" -", f)
    sys.exit(1)

print(
    f"Coverage citations OK ({checked} citations across {len(fns_cache)} test modules resolve to existing test fns)"
)
