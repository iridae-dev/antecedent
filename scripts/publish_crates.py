#!/usr/bin/env python3
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import tomllib

meta = json.loads(
    subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        text=True,
    )
)
workspace = {
    pkg["name"]: pkg
    for pkg in meta["packages"]
    if pkg["id"] in set(meta["workspace_members"])
}
members = {
    name: pkg
    for name, pkg in workspace.items()
    if pkg["name"] != "antecedent-py"
    # `publish = false` crates (antecedent-test-support) are dev-only.
    and pkg.get("publish") != []
}
deps = {name: set() for name in members}
version_errors = []
for name, pkg in members.items():
    for dep in pkg.get("dependencies", []):
        private = (
            dep["name"] in workspace and workspace[dep["name"]].get("publish") == []
        )
        if (
            dep["name"] in workspace
            and workspace[dep["name"]].get("publish") == []
            and (dep.get("kind") != "dev" or dep["req"] != "*")
        ):
            version_errors.append(
                f"{name}: unpublished dependency {dep['name']} must be a "
                "path-only dev dependency without a version"
            )
        if (dep["name"] == name or private) and dep.get("kind") == "dev":
            # A versioned self dependency survives Cargo packaging and tries to
            # resolve this not-yet-published crate from the registry. A path-only
            # self dependency is omitted from the published dev dependencies.
            manifest = tomllib.loads(
                Path(pkg["manifest_path"]).read_text(encoding="utf-8")
            )
            tables = [manifest.get("dev-dependencies", {})]
            tables.extend(
                target.get("dev-dependencies", {})
                for target in manifest.get("target", {}).values()
            )
            alias = dep.get("rename") or dep["name"]
            specifications = [table.get(alias) for table in tables]
            if any(
                isinstance(spec, dict) and spec.get("workspace")
                for spec in specifications
            ):
                root = Path(meta["workspace_root"]) / "Cargo.toml"
                specifications.append(
                    tomllib.loads(root.read_text(encoding="utf-8"))
                    .get("workspace", {})
                    .get("dependencies", {})
                    .get(alias)
                )
            versioned = any(
                isinstance(spec, str) or isinstance(spec, dict) and "version" in spec
                for spec in specifications
            )
            if versioned or dep["req"] != "*":
                version_errors.append(
                    f"{name}: versioned {'unpublished' if private else 'self'} dev dependency must be path-only; "
                    "remove its version before packaging"
                )
        if dep["name"] in members:
            expected = "^" + members[dep["name"]]["version"]
            # Cargo omits unversioned path-only dev dependencies from packages.
            unpublished_dev = dep.get("kind") == "dev" and dep["req"] == "*"
            if not unpublished_dev and dep["req"] != expected:
                version_errors.append(
                    f"{name}: {dep['name']} requires {dep['req']}; expected {expected}"
                )
        if dep["name"] in deps and dep.get("kind") in (None, "normal", "build"):
            deps[name].add(dep["name"])
if version_errors:
    sys.exit("internal dependency version mismatch:\n" + "\n".join(version_errors))

dependents = defaultdict(set)
indegree = {name: 0 for name in members}
for name, upstream in deps.items():
    for dep in upstream:
        dependents[dep].add(name)
        indegree[name] += 1

ready = sorted(name for name, degree in indegree.items() if degree == 0)
order = []
while ready:
    name = ready.pop(0)
    order.append(name)
    for dependent in sorted(dependents[name]):
        indegree[dependent] -= 1
        if indegree[dependent] == 0:
            ready.append(dependent)
            ready.sort()

if len(order) != len(members):
    sys.exit("crate publish graph has a cycle")
for name in order:
    print(name)
