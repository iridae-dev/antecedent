#!/usr/bin/env python3
"""Refresh the digests a completed review binds, after its text or artifacts changed.

    bind_review_digests.py record provenance/<id>.toml
    bind_review_digests.py fixture conformance/<family>/<name>/expected.json

``record`` rewrites each ``[[method_review.artifacts]]`` ``sha256`` and ``record_sha256``
in place.  ``fixture`` rewrites each ``oracle.reference_review.artifacts`` ``sha256`` and
``fixture_payload_sha256``.  The tool only binds what the author wrote; it does not decide
that a review is correct, and it never adds a review.  Run it after the last edit to any
bound artifact, then run the provenance and evidence gates.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT
from evidence_kinds import fixture_payload_digest
from provenance_links import method_review_digest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_record(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    record = tomllib.loads(text)
    review = record.get("method_review")
    if not isinstance(review, dict):
        raise SystemExit(f"{path}: no method_review to bind")
    for artifact in review["artifacts"]:
        rel = artifact["path"]
        digest = sha(REPO_ROOT / rel)
        pattern = re.compile(
            r'(path = "' + re.escape(rel) + r'"\nsha256 = ")[0-9a-f]{0,64}(")'
        )
        text, n = pattern.subn(r"\g<1>" + digest + r"\2", text)
        if n != 1:
            raise SystemExit(f"{path}: cannot locate digest line for {rel}")
    record = tomllib.loads(text)
    digest = method_review_digest(record)
    text, n = re.subn(
        r'(record_sha256 = ")[0-9a-f]{0,64}(")', r"\g<1>" + digest + r"\2", text
    )
    if n != 1:
        raise SystemExit(f"{path}: missing record_sha256 line")
    path.write_text(text, encoding="utf-8")


def bind_fixture(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    review = document["oracle"]["reference_review"]
    for artifact in review["artifacts"]:
        artifact["sha256"] = sha(REPO_ROOT / artifact["path"])
    review["fixture_payload_sha256"] = fixture_payload_digest(document)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"record", "fixture"}:
        print(__doc__)
        return 2
    path = (REPO_ROOT / sys.argv[2]).resolve()
    (bind_record if sys.argv[1] == "record" else bind_fixture)(path)
    print(f"bound {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
