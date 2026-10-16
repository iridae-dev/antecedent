"""Shared helpers for repository scripts.

``scripts/_repo.py`` is the import path existing scripts use (``python3
scripts/foo.py`` puts ``scripts/`` on ``sys.path``). This package re-exports
that module so new scripts can ``from _lib import read_text``. Environment
overrides such as ``INTERVAL_COORDINATES_ROOT`` stay in the check that owns
them; they are not a second copy of the repository root.
"""

from __future__ import annotations

from _repo import (
    REPO_ROOT,
    ensure_scripts_on_path,
    load_toml,
    read_text,
    workspace_package_version,
    write_text,
)

__all__ = [
    "REPO_ROOT",
    "ensure_scripts_on_path",
    "load_toml",
    "read_text",
    "workspace_package_version",
    "write_text",
]
