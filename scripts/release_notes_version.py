#!/usr/bin/env python3
"""Print the version the support-matrix generator writes notes for.

``docs/release-notes/preparation.toml`` is optional. When that file is present,
``target_version`` is the release being documented and may be ahead of the
package. When it is absent, the workspace package version in ``Cargo.toml`` is
the fallback. A missing file is not an error.
"""

from pathlib import Path

from _repo import load_toml_binary

notes = Path("docs/release-notes/preparation.toml")
if notes.is_file():
    data = load_toml_binary(notes)
else:
    data = load_toml_binary("Cargo.toml")["workspace"]["package"]
print(data.get("target_version", data.get("version")))
