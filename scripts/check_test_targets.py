#!/usr/bin/env python3
"""Every integration-test file is compiled by some test target.

A crate with `autotests = false` compiles only the files its `[[test]]` entries name and the
files a `suite_NN.rs` aggregator includes with `#[path]`; any other `tests/*.rs` file is
silently never built. This check lists such files. A `#[path]` include is followed
transitively, relative to the including file, so a file pulled in by another included file
(`joint_panel_workflow.rs` including `joint_panel_experiment_chain.rs`) counts as compiled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib
from _repo import REPO_ROOT as ROOT
from test_evidence import target_files


def included_files(root_file: Path) -> set[Path]:
    """`root_file` and every file reachable from it through `#[path]` includes."""
    root_file = root_file.resolve()
    crate = next((p for p in root_file.parents if (p / "Cargo.toml").is_file()), root_file.parent)
    boundary = crate.parent.parent if crate.parent.name == "crates" else crate
    return set(target_files(root_file, boundary))


def orphans_of(crate: Path) -> list[str]:
    data = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
    if data.get("package", {}).get("autotests", True):
        return []
    compiled: set[Path] = set()
    for entry in data.get("test", []):
        compiled |= included_files(crate / entry.get("path", f"tests/{entry['name']}.rs"))
    return [
        f"{crate.name}/tests/{f.name}"
        for f in sorted(crate.glob("tests/*.rs"))
        if f.resolve() not in compiled
    ]


def main() -> int:
    orphans = [
        o
        for manifest in sorted(ROOT.glob("crates/*/Cargo.toml"))
        for o in orphans_of(manifest.parent)
    ]
    for orphan in orphans:
        print(
            f"FAIL: crates/{orphan} is compiled by no test target; add it to a suite_NN.rs or a [[test]] entry"
        )
    if orphans:
        return 1
    print("test targets OK: every tests/*.rs file is compiled by a test target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
