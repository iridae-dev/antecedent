#!/usr/bin/env python3
"""Enforce production crate ownership, including optional and target dependencies.

Dev dependencies may exercise independent consumers. Production and build
dependencies must obey the same rules under every feature/platform selection.
"""
# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import sys
import tomllib
from collections import deque
from pathlib import Path

from _repo import REPO_ROOT


def dependency_graph(root: Path) -> dict[str, set[str]]:
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]
    inherited = workspace.get("dependencies", {})
    graph: dict[str, set[str]] = {}
    for member in workspace["members"]:
        for directory in root.glob(member):
            data = tomllib.loads((directory / "Cargo.toml").read_text(encoding="utf-8"))
            dependencies: set[str] = set()
            for table in [data, *data.get("target", {}).values()]:
                for kind in ("dependencies", "build-dependencies"):
                    for alias, spec in table.get(kind, {}).items():
                        spec = spec if isinstance(spec, dict) else {}
                        if spec.get("workspace"):
                            spec = inherited[alias]
                            spec = spec if isinstance(spec, dict) else {}
                        dependencies.add(spec.get("package", alias))
            graph[data["package"]["name"]] = dependencies
    return graph


def problems(graph: dict[str, set[str]]) -> list[str]:
    boundaries = {
        "antecedent-core": set(graph) - {"antecedent-core"},
        "antecedent-contract": {"antecedent-estimate", "antecedent-model", "antecedent"},
        "antecedent-expr": {"antecedent-estimate", "antecedent-model", "antecedent"},
        "antecedent-design": {"antecedent-estimate", "antecedent-validate", "antecedent-io", "antecedent"},
        "antecedent-io-core": {"antecedent-estimate", "antecedent-validate", "antecedent-io", "antecedent"},
        # Learners supply predictions; causal estimators and their facade consume them.
        "antecedent-learn": {"antecedent-estimate", "antecedent-validate", "antecedent"},
        "antecedent-learn-burn": {"antecedent-estimate", "antecedent-validate", "antecedent"},
    }
    found = []
    for owner, forbidden in boundaries.items():
        pending = deque([(owner, [owner])])
        seen = {owner}
        while pending:
            current, chain = pending.popleft()
            for dependency in sorted(graph.get(current, ())):
                if dependency in seen:
                    continue
                seen.add(dependency)
                path = [*chain, dependency]
                if dependency in forbidden:
                    found.append("ownership violation: " + " -> ".join(path))
                pending.append((dependency, path))
    return found


def main() -> int:
    failures = problems(dependency_graph(REPO_ROOT))
    for failure in failures:
        print(failure)
    if not failures:
        print("architecture OK: production crate dependency boundaries hold")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
