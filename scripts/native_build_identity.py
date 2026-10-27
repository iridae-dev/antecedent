#!/usr/bin/env python3
"""Record/check local native build inputs, including dirty sources and binary bytes.

A commit stamp alone misses edits in a working tree. Stamps are written only
after a successful build, never by the check operation. This is a conservative
build-input guard, not a claim of binary reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import platform
import sys
from pathlib import Path

from _repo import REPO_ROOT

STAMP = "target/native-build-identity/stamp.json"
PENDING = "target/native-build-identity/pending.json"


def inputs(root: Path) -> dict[str, str]:
    files = set()
    for pattern in (
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo/*.toml",
        "crates/*/Cargo.toml",
        "crates/*/build.rs",
        "crates/*/src/**/*",
        "crates/*/data/**/*",
        "python/Cargo.toml",
        "python/build.rs",
        "python/src/**/*",
        "python/pyproject.toml",
    ):
        files.update(p for p in root.glob(pattern) if p.is_file())
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def extension(root: Path) -> Path:
    # Follow this interpreter's extension-loader precedence. Another Python
    # ABI's cached extension may coexist without being importable here.
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = root / "python/antecedent" / f"_native{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError("no native extension matches this interpreter; rebuild it")


def identity(root: Path) -> dict:
    binary = extension(root)
    return {
        "schema": 1,
        "inputs": inputs(root),
        "filename": binary.name,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "python_abi": sys.implementation.cache_tag,
        "machine": platform.machine(),
        "system": platform.system(),
        "build_environment": {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RUSTUP_TOOLCHAIN",
                "CARGO_BUILD_TARGET",
            }
            or k.startswith("CARGO_PROFILE_RELEASE_")
        },
    }


def check(root: Path) -> None:
    stamp = root / STAMP
    if not stamp.is_file():
        raise ValueError(
            "native build identity missing; run bash scripts/test.sh python"
        )

    stored = json.loads(stamp.read_text(encoding="utf-8"))
    current = identity(root)
    if stored != current:
        changed = sorted(
            k
            for k in stored.get("inputs", {}).keys() | current["inputs"].keys()
            if stored.get("inputs", {}).get(k) != current["inputs"].get(k)
        )
        detail = (
            ", ".join(changed[:5]) or "extension bytes or interpreter/platform identity"
        )
        raise ValueError(
            f"native extension is stale ({detail}); run bash scripts/test.sh python"
        )


def begin_build(root: Path) -> None:
    (root / PENDING).parent.mkdir(parents=True, exist_ok=True)
    (root / PENDING).write_text(
        json.dumps(inputs(root), sort_keys=True), encoding="utf-8"
    )


def finish_build(root: Path) -> None:
    pending = root / PENDING
    if not pending.is_file():
        raise ValueError("native build input snapshot missing; begin before building")
    recorded_inputs = json.loads(pending.read_text(encoding="utf-8"))
    completed = identity(root)
    if recorded_inputs != completed["inputs"]:
        raise ValueError(
            "native inputs changed during the build; rebuild from stable sources"
        )
    (root / STAMP).write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    pending.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("begin", "stamp", "check"))
    args = parser.parse_args()
    try:
        if args.command == "begin":
            begin_build(REPO_ROOT)
        elif args.command == "stamp":
            finish_build(REPO_ROOT)
        else:
            check(REPO_ROOT)
    except (ValueError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"native build identity {args.command}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
