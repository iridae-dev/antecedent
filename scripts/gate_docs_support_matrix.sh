#!/usr/bin/env bash
# Docs vs support-matrix honesty: capabilities/comparison must point at the
# matrix and must not treat closed/n-a cells as licensed public analyze support.
#
# Run standalone or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

root = Path(".")
fail: list[str] = []

DOCS = ("docs/capabilities.md", "docs/comparison.md")
MATRIX_LINK = "support-matrix.md"

# Same-sentence / same-paragraph hedges: the claim is inventory or refusal,
# not a public analyze license.
HEDGE = re.compile(
    r"refus(?:e|es|ed)|not licensed|not a licensed|inventory|support matrix",
    re.I,
)

# Overclaims that read as licensed analyze support unless already hedged.
# Conservative: match the banned phrases only; do not scan every "PAG".
OVERCLAIMS = (
    re.compile(r"PAG\s+responses?", re.I),
    re.compile(r"PAG\s+curve", re.I),
    re.compile(r"we\s+support\s+PAG", re.I),
)


def paragraphs(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"\n[ \t]*\n", text):
        out.append((start, m.start(), text[start : m.start()]))
        start = m.end()
    out.append((start, len(text), text[start:]))
    return out


def sentences(para: str, para_start: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+", para):
        out.append((para_start + start, para_start + m.start(), para[start : m.start()]))
        start = m.end()
    out.append((para_start + start, para_start + len(para), para[start:]))
    return out


def containing(
    spans: list[tuple[int, int, str]], pos: int
) -> tuple[int, int, str] | None:
    for span in spans:
        if span[0] <= pos < span[1]:
            return span
    return None


for rel in DOCS:
    path = root / rel
    if not path.is_file():
        fail.append(f"{rel}: missing")
        continue
    text = path.read_text()
    if MATRIX_LINK not in text:
        fail.append(f"{rel}: does not link to {MATRIX_LINK}")

    paras = paragraphs(text)
    for pat in OVERCLAIMS:
        for m in pat.finditer(text):
            para_span = containing(paras, m.start())
            para = para_span[2] if para_span else ""
            sent = para
            if para_span is not None:
                sent_span = containing(sentences(para, para_span[0]), m.start())
                if sent_span is not None:
                    sent = sent_span[2]
            if HEDGE.search(sent) or HEDGE.search(para):
                continue
            snippet = re.sub(r"\s+", " ", m.group(0))
            fail.append(
                f"{rel}: overclaim {snippet!r} without refuse/not licensed/"
                f"inventory/support matrix nearby"
            )

# Cheap Counterfactual honesty check (capabilities only).
caps = root / "docs/capabilities.md"
if caps.is_file():
    text = caps.read_text()
    paras = paragraphs(text)
    hits = list(re.finditer(r"Counterfactual", text))
    if not hits:
        fail.append("docs/capabilities.md: Counterfactual is not mentioned")
    else:
        ok = False
        for m in hits:
            para_span = containing(paras, m.start())
            para = para_span[2] if para_span else ""
            window = text[max(0, m.start() - 240) : m.end() + 240]
            if HEDGE.search(para) or HEDGE.search(window):
                ok = True
                break
        if not ok:
            fail.append(
                "docs/capabilities.md: Counterfactual is not marked refused / "
                "not licensed nearby"
            )

# The current release notes are part of the 0.9 public contract, not an
# optional changelog paraphrase.  Verify that their refusal section accounts
# for every root query with zero licensed cells; otherwise a newly added root
# query (or a removed license) can leave the release notes silently stale.
cargo = tomllib.load(open(root / "Cargo.toml", "rb"))
version = cargo["workspace"]["package"]["version"]
notes_path = root / "docs" / "release-notes" / f"v{version}.md"
if not notes_path.is_file():
    fail.append(f"{notes_path}: current release notes missing")
else:
    notes = notes_path.read_text()
    if "../support-matrix.md" not in notes:
        fail.append(f"{notes_path}: does not link the authoritative support matrix")
    refusal_heading = "## Explicit refusals"
    if refusal_heading not in notes:
        fail.append(f"{notes_path}: missing {refusal_heading!r}")
    else:
        refusal_text = notes.split(refusal_heading, 1)[1].split("\n## ", 1)[0]
        axes = tomllib.load(open(root / "parity/support_axes.toml", "rb"))
        licensed = tomllib.load(open(root / "parity/support_licensed.toml", "rb")).get(
            "cell", []
        )
        licensed_queries = {row["query"] for row in licensed}
        zero_cell_queries = set(axes["queries"]) - licensed_queries
        for query in sorted(zero_cell_queries):
            if f"`{query}`" not in refusal_text:
                fail.append(
                    f"{notes_path}: zero-cell root query {query!r} is absent from "
                    "the explicit-refusals section"
                )

if fail:
    print("Docs support-matrix gate FAILED:")
    for item in fail:
        print(f" - {item}")
    sys.exit(1)

print(
    "Docs support-matrix OK (capabilities/comparison link the matrix; current "
    "release notes enumerate every zero-cell root query; no unhedged overclaims)"
)
PY
