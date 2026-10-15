#!/usr/bin/env python3
"""Every integration-test file is compiled by some test target.

A crate with `autotests = false` compiles only the files its `[[test]]` entries name and the
files a `suite_NN.rs` aggregator includes with `#[path]`; any other `tests/*.rs` file is
silently never built. This check lists such files.
"""

from __future__ import annotations

import re
import sys
import tomllib

from _repo import REPO_ROOT as ROOT

orphans = []
for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")):
    crate = manifest.parent
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    if data.get("package", {}).get("autotests", True):
        continue
    compiled = {entry.get("path", f"tests/{entry['name']}.rs") for entry in data.get("test", [])}
    for suite in crate.glob("tests/suite_*.rs"):
        compiled |= {f"tests/{p}" for p in re.findall(r'#\[path\s*=\s*"([^"]+)"\]', suite.read_text(encoding="utf-8"))}
    orphans += [f"{crate.name}/tests/{f.name}" for f in sorted(crate.glob("tests/*.rs")) if f"tests/{f.name}" not in compiled]
for orphan in orphans:
    print(f"FAIL: crates/{orphan} is compiled by no test target; add it to a suite_NN.rs or a [[test]] entry")
if orphans:
    sys.exit(1)
print("test targets OK: every tests/*.rs file is compiled by a test target")
