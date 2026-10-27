#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(".")
fail: list[str] = []

DOCS = ("docs/capabilities.md", "docs/comparison.md")
MATRIX_LINK = "support-matrix.md"

for rel in DOCS:
    path = root / rel
    if not path.is_file():
        fail.append(f"{rel}: missing")
        continue
    text = path.read_text(encoding="utf-8")
    if MATRIX_LINK not in text:
        fail.append(f"{rel}: does not link to {MATRIX_LINK}")

# The current release notes are part of the 0.9 public contract, not an
# optional changelog paraphrase.  Verify that their refusal section accounts
# for every root query with zero licensed cells; otherwise a newly added root
# query (or a removed license) can leave the release notes silently stale.
with open(root / "Cargo.toml", "rb") as handle:
    cargo = tomllib.load(handle)
version = cargo["workspace"]["package"]["version"]
# Optional. Absent means the workspace package version; only a present file
# with an invalid target_version is an error.
preparation = root / "docs" / "release-notes" / "preparation.toml"
if preparation.is_file():
    target = tomllib.loads(preparation.read_text(encoding="utf-8")).get("target_version")
    if not isinstance(target, str) or not re.fullmatch(r"\d+\.\d+\.\d+", target):
        fail.append("docs/release-notes/preparation.toml has no valid target_version")
        target = version
else:
    target = version
notes_path = root / "docs" / "release-notes" / f"v{target}.md"
if not notes_path.is_file():
    fail.append(f"{notes_path}: current release notes missing")
else:
    notes = notes_path.read_text(encoding="utf-8")
    if "../support-matrix.md" not in notes:
        fail.append(f"{notes_path}: does not link the authoritative support matrix")
    refusal_heading = "## Explicit refusals"
    if refusal_heading not in notes:
        fail.append(f"{notes_path}: missing {refusal_heading!r}")
    else:
        refusal_text = notes.split(refusal_heading, 1)[1].split("\n## ", 1)[0]
        with open(root / "parity/support_axes.toml", "rb") as handle:
            axes = tomllib.load(handle)
        with open(root / "parity/support_licensed.toml", "rb") as handle:
            licensed = tomllib.load(handle).get("cell", [])
        licensed_queries = {row["query"] for row in licensed}
        zero_cell_queries = set(axes["queries"]) - licensed_queries
        for query in sorted(zero_cell_queries):
            if f"`{query}`" not in refusal_text:
                fail.append(
                    f"{notes_path}: zero-cell root query {query!r} is absent from "
                    "the explicit-refusals section"
                )

generated = subprocess.run(
    [
        sys.executable,
        str(Path(__file__).resolve().parent / "generate_claim_inventory.py"),
        "--check",
    ],
    cwd=root,
    capture_output=True,
    text=True,
)
if generated.returncode != 0:
    detail = (generated.stdout + generated.stderr).strip()
    fail.extend(detail.splitlines() or ["generate_claim_inventory.py --check failed"])

if fail:
    print("Docs support-matrix gate FAILED:")
    for item in fail:
        print(f" - {item}")
    sys.exit(1)

print(
    "Docs support-matrix OK (capabilities/comparison link the matrix; current "
    "release notes enumerate every zero-cell root query; the per-query inventory "
    "and claim markers match the support registry and parity/claims.toml)"
)
