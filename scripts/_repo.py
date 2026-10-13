"""Repository root and UTF-8 text IO for scripts.

``Path.read_text()`` and ``Path.write_text()`` follow the process locale, and
``write_text`` translates ``\\n`` to ``os.linesep``. Windows cp1252 cannot
round-trip the support-matrix times sign. Go through these helpers (or pass
``encoding="utf-8"``) when a script reads or writes repository text.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_scripts_on_path() -> None:
    """Make sibling scripts importable.

    ``python3 scripts/foo.py`` already puts ``scripts/`` on ``sys.path``. An
    import of that module from another directory does not.
    """
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def read_text(path: Path | str, *, errors: str = "strict") -> str:
    return Path(path).read_text(encoding="utf-8", errors=errors)


def write_text(path: Path | str, data: str) -> int:
    return Path(path).write_text(data, encoding="utf-8", newline="\n")


def load_toml(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(read_text(path))


def cargo_target_feature_args(manifest: Path | str, target_args: list[str] | tuple[str, ...]) -> list[str]:
    """Return only the declared required features for one selected Cargo target.

    Cargo skips a target with ``required-features`` unless the matching feature
    set is enabled. Evidence listing and execution both use this helper so they
    discover and run the same target without enabling unrelated features.
    """
    args = list(target_args)
    target_kind: str | None = None
    target_name: str | None = None
    for switch, kind in (("--test", "test"), ("--bench", "bench"), ("--bin", "bin"), ("--example", "example")):
        if switch in args:
            index = args.index(switch)
            if index + 1 < len(args):
                target_kind = kind
                target_name = args[index + 1]
            break
    if target_kind is None and "--lib" in args:
        target_kind = "lib"

    if target_kind is None:
        return []

    cargo = load_toml(manifest)
    if target_kind == "lib":
        targets = [cargo.get("lib")] if isinstance(cargo.get("lib"), dict) else []
    else:
        targets = cargo.get(target_kind, [])
        if not isinstance(targets, list):
            targets = []
        targets = [target for target in targets if isinstance(target, dict) and target.get("name") == target_name]
    if len(targets) > 1:
        raise ValueError(f"{manifest}: duplicate Cargo {target_kind} target {target_name!r}")
    if not targets:
        return []

    required = targets[0].get("required-features", [])
    if not isinstance(required, list) or not all(isinstance(feature, str) and feature for feature in required):
        raise ValueError(f"{manifest}: invalid required-features for {target_kind} target {target_name!r}")
    if not required:
        return []
    return ["--features", ",".join(required)]


def workspace_package_version(root: Path | None = None) -> str:
    """``[workspace.package] version`` from the repo ``Cargo.toml``."""
    cargo = load_toml((root or REPO_ROOT) / "Cargo.toml")
    version = cargo["workspace"]["package"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("Cargo.toml workspace.package.version is missing")
    return version
