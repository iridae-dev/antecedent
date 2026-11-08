"""The manifest-only source of the cached Nix dependency build.

The tree holds the workspace manifests, Cargo.lock, the workspace Cargo
configuration and a stub for every workspace target. Third-party dependency
compilation depends only on these, so the derivation built from it
(`nix/evidence-deps.nix`) stays the same across commits that change no manifest.
Its stub workspace units are discarded; the evidence build always compiles the
real workspace source.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# Files Cargo reads to resolve and build dependencies, besides the member manifests.
WORKSPACE_FILES = ("Cargo.toml", "Cargo.lock", ".cargo/config.toml", ".cargo/config")
_LIB_KINDS = {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}


def write_deps_source(snapshot: Path, destination: Path) -> None:
    """Write the manifest-only tree of a verified committed `snapshot` into `destination`."""
    snapshot = snapshot.resolve()
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version=1", "--no-deps", "--offline", "--locked"],
        cwd=snapshot, text=True))
    destination.mkdir(parents=True, exist_ok=False)
    for rel in WORKSPACE_FILES:
        if (snapshot / rel).is_file():
            _copy(snapshot, destination, rel)
    for package in metadata["packages"]:
        _copy(snapshot, destination, _relative(snapshot, package["manifest_path"]))
        for target in package["targets"]:
            rel = _relative(snapshot, target["src_path"])
            stub = destination / rel
            if stub.exists():
                continue
            stub.parent.mkdir(parents=True, exist_ok=True)
            kinds = set(target["kind"])
            # A library compiles empty; every other target needs an entry point
            # (a test harness ignores it, a harness-free target calls it).
            stub.write_text("" if kinds & _LIB_KINDS else "#[allow(dead_code)]\nfn main() {}\n",
                            encoding="utf-8")
    for path in sorted(destination.rglob("*")):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def _relative(snapshot: Path, path: str) -> str:
    rel = Path(path).resolve().relative_to(snapshot)
    if ".." in rel.parts:
        raise ValueError("workspace path leaves the committed snapshot")
    return rel.as_posix()


def _copy(snapshot: Path, destination: Path, rel: str) -> None:
    target = destination / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot / rel, target)


def import_deps_source(tree: Path, *, run=subprocess.run) -> str:
    """Add the tree to the store by content: identical manifests give the same path."""
    result = run(["nix", "--extra-experimental-features", "nix-command", "--offline", "store", "add-path",
                  "--name", "antecedent-deps-source", str(tree)],
                 check=True, capture_output=True, text=True)
    return result.stdout.strip()
