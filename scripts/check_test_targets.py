#!/usr/bin/env python3
"""Every integration-test file is compiled by some test target.

A crate with `autotests = false` compiles only the files its `[[test]]` entries name and the
files a `suite_NN.rs` aggregator includes with `#[path]`; any other `tests/*.rs` file is
silently never built. This check lists such files. A `#[path]` include is followed
transitively, relative to the including file, so a file pulled in by another included file
(`joint_panel_workflow.rs` including `joint_panel_experiment_chain.rs`) counts as compiled.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as ROOT

PATH_ATTR = re.compile(r'#\[path\s*=\s*"([^"]+)"\]')


def included_files(root_file: Path) -> set[Path]:
    """`root_file` and every file reachable from it through `#[path]` includes."""
    seen: set[Path] = set()
    pending = [root_file.resolve()]
    while pending:
        current = pending.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        for target in PATH_ATTR.findall(current.read_text(encoding="utf-8")):
            pending.append((current.parent / target).resolve())
    return seen


def orphans_of(crate: Path) -> list[str]:
    data = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
    if data.get("package", {}).get("autotests", True):
        return []
    compiled: set[Path] = set()
    for entry in data.get("test", []):
        compiled |= included_files(crate / entry.get("path", f"tests/{entry['name']}.rs"))
    for suite in crate.glob("tests/suite_*.rs"):
        compiled |= included_files(suite)
    return [f"{crate.name}/tests/{f.name}" for f in sorted(crate.glob("tests/*.rs")) if f.resolve() not in compiled]


def main() -> int:
    orphans = [o for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")) for o in orphans_of(manifest.parent)]
    for orphan in orphans:
        print(f"FAIL: crates/{orphan} is compiled by no test target; add it to a suite_NN.rs or a [[test]] entry")
    if orphans:
        return 1
    print("test targets OK: every tests/*.rs file is compiled by a test target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
