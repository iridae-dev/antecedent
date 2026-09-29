#!/usr/bin/env python3
"""Every Criterion bench target of the workspace, from `cargo metadata`.

    python3 scripts/bench_targets.py            # one `<package> <bench>` per line

The release gate smoke-runs every target in one `cargo bench --workspace` invocation
(`--exclude` only for members with no benches) on main and on any pull request
that touches `benches/` or a `[[bench]]` entry. This script is that member list:
a bench added under an excluded package fails the gate until the exclude is lifted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bench_targets() -> list[tuple[str, str]]:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    members = set(meta["workspace_members"])
    return sorted(
        (pkg["name"], target["name"])
        for pkg in meta["packages"]
        if pkg["id"] in members
        for target in pkg["targets"]
        if "bench" in target["kind"]
    )


if __name__ == "__main__":
    targets = bench_targets()
    if not targets:
        sys.exit("cargo metadata lists no bench targets")
    for package, bench in targets:
        print(package, bench)
