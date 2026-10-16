#!/usr/bin/env python3
"""Print the version the support-matrix generator writes notes for.

``docs/release-notes/preparation.toml`` is optional. When that file is present,
``target_version`` is the release being documented and may be ahead of the
package. When it is absent, the workspace package version in ``Cargo.toml`` is
the fallback. A missing file is not an error.
"""

import tomllib
from pathlib import Path

notes = Path("docs/release-notes/preparation.toml")
if notes.is_file():
    with notes.open("rb") as handle:
        data = tomllib.load(handle)
else:
    with Path("Cargo.toml").open("rb") as handle:
        data = tomllib.load(handle)["workspace"]["package"]
print(data.get("target_version", data.get("version")))
