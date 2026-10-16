#!/usr/bin/env python3
import json
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

meta = json.loads(
    subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        text=True,
    )
)
members = {
    pkg["name"]: pkg
    for pkg in meta["packages"]
    if pkg["id"] in set(meta["workspace_members"])
    and pkg["name"] != "antecedent-py"
    # `publish = false` crates (antecedent-test-support) are dev-only.
    and pkg.get("publish") != []
}
deps = {name: set() for name in members}
version_errors = []
for name, pkg in members.items():
    for dep in pkg.get("dependencies", []):
        if dep["name"] == name and dep.get("kind") == "dev":
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
            alias = dep.get("rename") or name
            versioned = any(
                isinstance(table.get(alias), str)
                or isinstance(table.get(alias), dict)
                and "version" in table[alias]
                for table in tables
            )
            if versioned or dep["req"] != "*":
                version_errors.append(
                    f"{name}: versioned self dev dependency must be path-only; "
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
