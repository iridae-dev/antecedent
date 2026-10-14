#!/usr/bin/env python3
"""Stage or install the same-run Python extension used by Linux CI checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from pathlib import Path

MANIFEST = "native-build.json"


def runtime_identity() -> dict[str, object]:
    return {
        "implementation": sys.implementation.name,
        "python": list(sys.version_info[:2]),
        "platform": sys.platform,
        "machine": platform.machine(),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
    }


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package_version(root: Path) -> str:
    with (root / "python" / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def verify_import(root: Path, extension: Path, version: str) -> None:
    sys.path.insert(0, str(root / "python"))
    native = importlib.import_module("antecedent._native")
    if Path(native.__file__).resolve() != extension.resolve():
        raise ValueError(f"import resolved to {native.__file__}, expected {extension}")
    if native.__version__ != version:
        raise ValueError(f"native version {native.__version__} != checkout version {version}")


def verify_fresh_import(extension: Path, version: str) -> None:
    # Several facade tests spawn Python from a temporary working directory.
    # Exercise that same import path without inserting the checkout into sys.path.
    code = (
        "from pathlib import Path; import sys; import antecedent._native as native; "
        "assert Path(native.__file__).resolve() == Path(sys.argv[1]).resolve(), native.__file__; "
        "assert native.__version__ == sys.argv[2], native.__version__"
    )
    with tempfile.TemporaryDirectory() as outside_checkout:
        proc = subprocess.run(
            [sys.executable, "-c", code, str(extension.resolve()), version],
            cwd=outside_checkout,
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode:
        raise ValueError(
            f"fresh-process native import failed: {(proc.stdout + proc.stderr)[-2000:]}"
        )


def stage(root: Path, artifact: Path, commit: str) -> None:
    runtime = runtime_identity()
    extension = root / "python" / "antecedent" / f"_native{runtime['extension_suffix']}"
    version = package_version(root)
    verify_import(root, extension, version)
    artifact.mkdir(parents=True, exist_ok=True)
    shutil.copy2(extension, artifact / extension.name)
    manifest = {
        "commit": commit,
        "runtime": runtime,
        "filename": extension.name,
        "sha256": digest(extension),
        "version": version,
    }
    (artifact / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"staged {extension.name} for {commit}")


def install(root: Path, artifact: Path, commit: str) -> None:
    manifest = json.loads((artifact / MANIFEST).read_text(encoding="utf-8"))
    runtime = runtime_identity()
    version = package_version(root)
    for key, expected in (("commit", commit), ("runtime", runtime), ("version", version)):
        if manifest.get(key) != expected:
            raise ValueError(f"native artifact {key} {manifest.get(key)!r} != {expected!r}")
    filename = f"_native{runtime['extension_suffix']}"
    if manifest.get("filename") != filename:
        raise ValueError(f"native artifact filename {manifest.get('filename')!r} != {filename!r}")
    source = artifact / filename
    if digest(source) != manifest.get("sha256"):
        raise ValueError("native artifact checksum mismatch")
    destination = root / "python" / "antecedent" / filename
    shutil.copy2(source, destination)
    # Match editable-install semantics without recompiling. This persists the
    # checkout package path for pytest and the child Python processes it starts.
    site_packages = Path(sysconfig.get_path("purelib"))
    (site_packages / "antecedent-ci-source.pth").write_text(
        str((root / "python").resolve()) + "\n", encoding="utf-8"
    )
    verify_fresh_import(destination, version)
    print(f"installed and imported {filename} for {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "install"))
    parser.add_argument("artifact", type=Path)
    parser.add_argument("commit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        {"stage": stage, "install": install}[args.operation](root, args.artifact, args.commit)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"native artifact: {error}\n")


if __name__ == "__main__":
    main()
