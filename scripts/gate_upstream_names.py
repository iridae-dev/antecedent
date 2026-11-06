#!/usr/bin/env python3
"""Fail when a dependency manifest declares an upstream package.

Reads dependency tables only: `Cargo.toml` dependency sections and
`pyproject.toml` dependency lists/groups, plus `requirements*.txt` requirement
names. Comments, docstrings, documentation and free-text fields are not read.
The banned names live in parity/permitted_upstream_name_files.toml (`names`);
`exact` lists manifests that may declare them.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path
import _repo

root = Path(".")
permitted_path = root / "parity/permitted_upstream_name_files.toml"
CARGO_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def fail(message: str) -> None:
    print(message)
    sys.exit(1)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_spec(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        fail(f"upstream-name permitted files: missing {path}")
    try:
        data = _repo.load_toml(path)
    except tomllib.TOMLDecodeError as exc:
        fail(f"upstream-name permitted files: {path} is not valid TOML: {exc}")
    out = []
    for key in ("names", "exact"):
        value = data.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            fail(f"upstream-name permitted files: `{key}` must be a list of non-empty strings")
        out.append(set(value))
    if not out[0]:
        fail(f"upstream-name permitted files: {path} missing `names`")
    return {norm(n) for n in out[0]}, out[1]


def requirement_name(spec: str) -> str | None:
    match = REQ_NAME.match(spec)
    return norm(match.group(1)) if match else None


def cargo_dependency_names(data: dict) -> set[str]:
    names: set[str] = set()
    tables = [data]
    tables.extend(t for t in data.get("target", {}).values() if isinstance(t, dict))
    workspace = data.get("workspace")
    if isinstance(workspace, dict):
        tables.append(workspace)
    for table in tables:
        for section in CARGO_TABLES:
            deps = table.get(section, {})
            if not isinstance(deps, dict):
                continue
            for key, value in deps.items():
                names.add(norm(key))
                if isinstance(value, dict) and isinstance(value.get("package"), str):
                    names.add(norm(value["package"]))
    return names


def pyproject_dependency_names(data: dict) -> set[str]:
    specs: list[str] = []
    project = data.get("project", {})
    specs.extend(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    for group in data.get("dependency-groups", {}).values():
        specs.extend(item for item in group if isinstance(item, str))
    build = data.get("build-system", {})
    specs.extend(build.get("requires", []))
    return {n for n in map(requirement_name, specs) if n}


def manifest_names(path: Path) -> set[str]:
    if path.name == "Cargo.toml":
        return cargo_dependency_names(_repo.load_toml(path))
    if path.name == "pyproject.toml":
        return pyproject_dependency_names(_repo.load_toml(path))
    if path.name.startswith("requirements") and path.suffix == ".txt":
        names = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                name = requirement_name(line)
                if name:
                    names.add(name)
        return names
    return set()


banned, exact = load_spec(permitted_path)
tracked = subprocess.run(
    ["git", "ls-files", "-z"], check=True, capture_output=True
).stdout.split(b"\0")

hits = []
for raw in tracked:
    if not raw:
        continue
    path = root / raw.decode()
    if not path.is_file() or path.as_posix() in exact:
        continue
    if any(part.startswith(".") and part != ".github" for part in path.parts):
        continue
    if not (
        path.name in ("Cargo.toml", "pyproject.toml")
        or (path.name.startswith("requirements") and path.suffix == ".txt")
    ):
        continue
    try:
        found = manifest_names(path) & banned
    except (OSError, tomllib.TOMLDecodeError) as exc:
        hits.append((path.as_posix(), [f"unreadable manifest: {exc}"]))
        continue
    if found:
        hits.append((path.as_posix(), sorted(found)))

if hits:
    print(f"upstream-name violations ({len(hits)} manifests):")
    for path, found in hits:
        print(f"- {path}: {', '.join(found)}")
    sys.exit(1)
print("upstream-name permitted files: ok")
