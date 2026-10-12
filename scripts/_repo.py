"""Repository root and UTF-8 text IO for scripts.

``Path.read_text()`` and ``Path.write_text()`` follow the process locale, and
``write_text`` translates ``\\n`` to ``os.linesep``. Windows cp1252 cannot
round-trip the support-matrix times sign. Go through these helpers (or pass
``encoding="utf-8"``) when a script reads or writes repository text.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Path | str, data: str) -> int:
    return Path(path).write_text(data, encoding="utf-8", newline="\n")


def load_toml(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(read_text(path))


def workspace_package_version(root: Path | None = None) -> str:
    """``[workspace.package] version`` from the repo ``Cargo.toml``."""
    cargo = load_toml((root or REPO_ROOT) / "Cargo.toml")
    version = cargo["workspace"]["package"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("Cargo.toml workspace.package.version is missing")
    return version
