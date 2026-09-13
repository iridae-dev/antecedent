"""UTF-8 loaders for repository text.

Windows `Path.read_text()` defaults to the locale encoding (cp1252). Conformance
pins use U+00D7 (`×`) in support-matrix cell names. Decoding those files as
cp1252 turns `×` into replacement characters and fails only on Windows wheels.

Always go through these helpers (or pass `encoding="utf-8"`) when reading
repo JSON, TOML, Markdown, or other text from tests.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: Path | str) -> Any:
    return json.loads(read_text(path))


def load_toml(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(read_text(path))
