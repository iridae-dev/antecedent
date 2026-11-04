#!/usr/bin/env python3
"""Keep split Rust CI suites on the full workspace's Cargo feature graph."""

import json
import re
import subprocess
import sys
from pathlib import Path

CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"


def workflow_features() -> tuple[str, str]:
    """The two feature lists the split Rust suites pass, read from the ci workflow's env."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    found = []
    for name in ("RUST_CI_REST_FEATURES", "RUST_CI_FACADE_FEATURES"):
        match = re.search(rf"^\s*{name}:\s*(\S+)\s*$", text, re.M)
        if match is None:
            raise SystemExit(f"{CI_WORKFLOW}: {name} is not set")
        found.append(match.group(1))
    return found[0], found[1]


def cargo(*args: str) -> str:
    return subprocess.check_output(["cargo", *args], text=True)


def main() -> None:
    if len(sys.argv) == 1:
        rest_features, facade_features = workflow_features()
    elif len(sys.argv) == 3:
        rest_features, facade_features = sys.argv[1:]
    else:
        raise SystemExit("usage: check_rust_ci_feature_parity.py [REST_FEATURES FACADE_FEATURES]")
    metadata = json.loads(cargo("metadata", "--no-deps", "--format-version", "1"))
    workspace = set(metadata["workspace_members"])
    names = {package["name"] for package in metadata["packages"] if package["id"] in workspace}

    def features(*selection: str) -> dict[str, set[str]]:
        lines = cargo(
            "tree",
            *selection,
            "-e",
            "normal,build,dev",
            "--prefix",
            "none",
            "--format",
            "{p} [{f}]",
        ).splitlines()
        found: dict[str, set[str]] = {}
        for line in lines:
            package, separator, rest = line.partition(" v")
            if not separator or package not in names or " [" not in rest:
                continue
            enabled = rest.rsplit(" [", 1)[1].split("]", 1)[0]
            found.setdefault(package, set()).update(enabled.split(",") if enabled else ())
        return found

    full = features("--workspace")
    shards = {
        "rest": features("--workspace", "--exclude", "antecedent", "--features", rest_features),
        "facade": features("-p", "antecedent", "--features", facade_features),
    }
    expected = {"rest": names - {"antecedent"}, "facade": {"antecedent"}}
    problems = []
    for shard, graph in shards.items():
        for package in sorted(expected[shard]):
            if package not in graph:
                problems.append(f"{shard}: {package} is absent from the Cargo graph")
        for package, enabled in sorted(graph.items()):
            if enabled != full.get(package):
                problems.append(
                    f"{shard}: {package} features {sorted(enabled)} differ from full workspace "
                    f"{sorted(full.get(package, set()))}"
                )
    if problems:
        raise SystemExit("Rust CI feature parity failed:\n" + "\n".join(problems))
    print(f"Rust CI feature parity: ok ({len(names)} workspace packages, two suites)")


if __name__ == "__main__":
    main()
