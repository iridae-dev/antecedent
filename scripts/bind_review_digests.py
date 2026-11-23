#!/usr/bin/env python3
"""Refresh the digests a completed review binds, after its text or artifacts changed.

    bind_review_digests.py record provenance/<id>.toml
    bind_review_digests.py mapping provenance/<id>.toml
    bind_review_digests.py fixture conformance/<family>/<name>/expected.json

``record`` rewrites each ``[[method_review.artifacts]]`` ``sha256`` and ``record_sha256``
in place.  ``mapping`` does the same for ``[[non_algorithm_review.artifacts]]`` and its ``record_sha256``.
``fixture`` rewrites each reference-review artifact ``sha256`` and ``fixture_payload_sha256``
and keeps the review in ``parity/fixture_reviews/<family>/<name>.json``, moving one still
embedded in the fixture (``oracle.reference_review``) out of it.

A source or test artifact binds the items the review covers (``items = ["fn solve", "impl
Kernel"]``, see ``review_items.py``): the digest hashes only those items, so an edit
elsewhere in the file leaves the review bound.  A data artifact binds the whole file.  The
author names the items; the tool only hashes them.  It does not decide that a review is
correct, it never adds a review, and a missing item is an error.  Rebinding states that the
reviewer re-read the changed items.  Run it after the last edit to any bound item, then run
the provenance and evidence gates.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT
from evidence_kinds import fixture_payload_digest, load_fixture, review_path
from provenance_links import method_review_digest, non_algorithm_digest
from review_items import artifact_digest


def bound_digest(artifact: dict) -> str:
    """The digest an artifact binds: its named items, or the whole file of a data artifact."""
    digest, error = artifact_digest(REPO_ROOT, artifact)
    if error is not None:
        raise SystemExit(f"{artifact.get('path')}: {error}")
    assert digest is not None
    return digest


def bind_artifact_digest(text: str, rel: str, digest: str) -> str:
    """Update one artifact in either multiline or inline TOML table form."""
    pattern = re.compile(
        r'(path = "'
        + re.escape(rel)
        + r'"(?:\n|, )(?:items = \[[^\]]*\](?:\n|, ))?sha256 = ")[0-9a-f]{0,64}(")'
    )
    text, count = pattern.subn(r"\g<1>" + digest + r"\2", text)
    if count != 1:
        raise SystemExit(f"cannot locate unique digest line for {rel}")
    return text


def bind_record(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    record = tomllib.loads(text)
    review = record.get("method_review")
    if not isinstance(review, dict):
        raise SystemExit(f"{path}: no method_review to bind")
    for artifact in review["artifacts"]:
        text = bind_artifact_digest(text, artifact["path"], bound_digest(artifact))
    record = tomllib.loads(text)
    digest = method_review_digest(record)
    text, n = re.subn(
        r'(record_sha256 = ")[0-9a-f]{0,64}(")', r"\g<1>" + digest + r"\2", text
    )
    if n != 1:
        raise SystemExit(f"{path}: missing record_sha256 line")
    path.write_text(text, encoding="utf-8")


def bind_mapping(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    record = tomllib.loads(text)
    review = record.get("non_algorithm_review")
    if not isinstance(review, dict):
        raise SystemExit(f"{path}: no non_algorithm_review to bind")
    for artifact in review["artifacts"]:
        text = bind_artifact_digest(text, artifact["path"], bound_digest(artifact))
    digest = non_algorithm_digest(tomllib.loads(text))
    text, n = re.subn(
        r'(record_sha256 = ")[0-9a-f]{0,64}(")', r"\g<1>" + digest + r"\2", text
    )
    if n != 1:
        raise SystemExit(f"{path}: missing record_sha256 line")
    path.write_text(text, encoding="utf-8")


def bind_fixture(path: Path) -> None:
    """Rebind a fixture's review and keep it in `parity/fixture_reviews`: a review still
    embedded in the fixture moves out, so this and later rebinds leave the fixture, which
    tests read, unchanged."""
    document = load_fixture(REPO_ROOT, path)
    review = document["oracle"]["reference_review"]
    for artifact in review["artifacts"]:
        artifact["sha256"] = bound_digest(artifact)
    review["fixture_payload_sha256"] = fixture_payload_digest(document)
    target = review_path(REPO_ROOT, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if "reference_review" in stored.get("oracle", {}):
        del stored["oracle"]["reference_review"]
        path.write_text(json.dumps(stored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"record", "mapping", "fixture"}:
        print(__doc__)
        return 2
    path = (REPO_ROOT / sys.argv[2]).resolve()
    {"record": bind_record, "mapping": bind_mapping, "fixture": bind_fixture}[sys.argv[1]](path)
    print(f"bound {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
