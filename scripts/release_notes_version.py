#!/usr/bin/env python3
"""Print the version the support-matrix generator writes notes for."""

from pathlib import Path

import tomllib

notes = Path("docs/release-notes/preparation.toml")
if notes.is_file():
    with notes.open("rb") as handle:
        data = tomllib.load(handle)
else:
    with Path("Cargo.toml").open("rb") as handle:
        data = tomllib.load(handle)["workspace"]["package"]
print(data.get("target_version", data.get("version")))
