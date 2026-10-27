#!/usr/bin/env python3
"""Self-test for snapshot_promotion_baseline.py (run with python3 -m unittest)."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_promotion_baseline as snap  # noqa: E402

REGISTRY_OLD = """version = 1
release = "0.9"
[[record]]
id = "old.record"
status = "promoted"
routes = []
"""

REGISTRY_TAGGED = """version = 1
release = "1.0"
[[record]]
id = "kept.record"
status = "promoted"
routes = [
  { name = "pkg.open_route", stage = "evaluate", status = "licensed", claim = "point_only" },
  { name = "pkg.refused_route", stage = "uncertainty", status = "closed", claim = "none", reason_code = "cell_not_licensed" },
]
[[record]]
id = "carried.record"
status = "carried_forward"
routes = [
  { name = "pkg.carried_route", stage = "identify", status = "closed", claim = "point_only", reason_code = "route_not_supported" },
]
"""

STAGES_TAGGED = """version = 1
default = "closed"
[[routes]]
route = "pkg.open_route"
stage = "evaluate"
status = "licensed"
[[routes]]
route = "pkg.refused_route"
stage = "uncertainty"
status = "closed"
reason_code = "cell_not_licensed"
[[routes]]
route = "stage.only_route"
stage = "prepare"
status = "closed"
reason_code = "route_not_supported"
"""


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def put(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(repo: Path, *argv: str) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = snap.run(["--repo", str(repo), *argv])
    return code, err.getvalue()


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        git(self.repo, "init", "-q")
        # An older release, then the release under test, then later drift.
        put(self.repo, "parity/promotion_old.toml", REGISTRY_OLD)
        put(self.repo, "parity/transport_stages.toml", 'version = 1\ndefault = "closed"\n')
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "old")
        git(self.repo, "tag", "v0.9.0")
        put(self.repo, "parity/promotion_cur.toml", REGISTRY_TAGGED)
        put(self.repo, "parity/transport_stages.toml", STAGES_TAGGED)
        put(self.repo, "parity/support_closed.toml", '[[closed]]\nreason = "r"\n')
        put(self.repo, "docs/x-producer-consumer-matrix.md", "matrix at tag\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "tagged")
        git(self.repo, "tag", "v1.0.0")
        git(self.repo, "tag", "v1.10.0")  # same commit; numeric order beats lexical
        git(self.repo, "tag", "v1.9.0")
        self.tag_commit = git(self.repo, "rev-parse", "HEAD")
        # Drift after the tag: working tree and a later commit both differ.
        put(
            self.repo,
            "parity/promotion.toml",
            'version = 1\n[[record]]\nid = "new"\nstatus = "frozen"\nroutes = []\n',
        )
        put(self.repo, "docs/x-producer-consumer-matrix.md", "matrix after tag\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "refactor")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def write(self) -> dict:
        code, err = run(self.repo, "--write", "--tag", "v1.0.0")
        self.assertEqual(code, 0, err)
        return tomllib.loads((self.repo / snap.SNAPSHOT_TOML).read_text(encoding="utf-8"))

    def test_latest_tag_uses_numeric_version_order(self) -> None:
        self.assertEqual(snap.latest_tag(self.repo), "v1.10.0")

    def test_snapshot_reads_tag_content_not_working_tree(self) -> None:
        # Edit the working tree after the base commit: the snapshot must not move.
        put(self.repo, "parity/transport_stages.toml", "version = 1\n")
        doc = self.write()
        self.assertEqual(doc["snapshot"]["tag_commit"], self.tag_commit)
        self.assertEqual(doc["snapshot"]["base_commit"], self.base)
        self.assertEqual(doc["snapshot"]["registry_path"], "parity/promotion_cur.toml")
        self.assertEqual(doc["snapshot"]["base_registry_path"], "parity/promotion_cur.toml")
        names = [r["route"] for r in doc["route"]]
        self.assertEqual(names, ["pkg.carried_route", "pkg.refused_route", "stage.only_route"])
        carried = doc["route"][0]
        self.assertTrue(carried["carryover"])
        self.assertEqual(carried["record"], "carried.record")
        self.assertEqual(carried["reason_code"], "route_not_supported")
        self.assertEqual(doc["route"][1]["sources"], ["promotion_registry", "stage_registry"])
        self.assertTrue(all(r["disposition"] == "unclassified" for r in doc["route"]))
        files = {f["role"]: f for f in doc["file"]}
        self.assertNotEqual(
            files["producer_consumer_matrix"]["tag_sha256"],
            files["producer_consumer_matrix"]["base_sha256"],
        )
        self.assertEqual(files["producer_consumer_registry"]["tag_sha256"], snap.ABSENT)
        self.assertEqual(doc["counts"]["inherited_closed_routes"], 3)

    def test_write_is_idempotent_and_check_passes(self) -> None:
        self.write()
        first = (self.repo / snap.SNAPSHOT_MD).read_bytes()
        self.write()
        self.assertEqual(first, (self.repo / snap.SNAPSHOT_MD).read_bytes())
        code, err = run(self.repo, "--check")
        self.assertEqual(code, 0, err)

    def test_check_rejects_a_hand_edited_snapshot(self) -> None:
        self.write()
        md = self.repo / snap.SNAPSHOT_MD
        md.write_text(md.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
        code, err = run(self.repo, "--check")
        self.assertEqual(code, 1)
        self.assertIn("does not match git", err)

    def test_classification_survives_write_and_is_validated(self) -> None:
        doc = self.write()
        toml_path = self.repo / snap.SNAPSHOT_TOML
        text = toml_path.read_text(encoding="utf-8")
        target = 'route = "pkg.refused_route"'
        head, tail = text.split(target, 1)
        tail = tail.replace('disposition = "unclassified"', 'disposition = "deliberate_refusal"', 1)
        tail = tail.replace("touched_by = []", 'touched_by = ["composition"]', 1)
        tail = tail.replace('rationale = ""', 'rationale = "kept refusing"', 1)
        toml_path.write_text(head + target + tail, encoding="utf-8")
        code, err = run(self.repo, "--write")
        self.assertEqual(code, 0, err)
        again = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        route = next(r for r in again["route"] if r["route"] == "pkg.refused_route")
        self.assertEqual(route["disposition"], "deliberate_refusal")
        self.assertEqual(route["reason_code"], "cell_not_licensed")
        self.assertEqual(route["rationale"], "kept refusing")
        self.assertEqual(len(again["route"]), len(doc["route"]))
        code, err = run(self.repo, "--check")
        self.assertEqual(code, 0, err)

    def test_malformed_classification_is_an_error(self) -> None:
        doc = self.write()
        route = doc["route"][0]
        route["disposition"] = "reopened_under_new_record"
        route["touched_by"] = ["composition"]
        route["rationale"] = "why"
        errors, _ = snap.validate(doc, self.repo, strict=False)
        self.assertTrue(any("requires new_record" in e for e in errors))
        route["new_record"] = "not.allocated"
        errors, warnings = snap.validate(doc, self.repo, strict=False)
        self.assertEqual(errors, [])
        self.assertTrue(any("not.allocated" in w for w in warnings))
        errors, _ = snap.validate(doc, self.repo, strict=True)
        self.assertTrue(any("not.allocated" in e for e in errors))
        route["disposition"] = "prerequisite"
        errors, _ = snap.validate(doc, self.repo, strict=False)
        self.assertTrue(any("only valid for reopened" in e for e in errors))
        route["disposition"] = "nonsense"
        errors, _ = snap.validate(doc, self.repo, strict=False)
        self.assertTrue(any("unknown disposition" in e for e in errors))

    def test_decision_pointer_must_resolve(self) -> None:
        doc = self.write()
        put(self.repo, "NOTES.md", "identity moved to the canonical writer\n")
        doc["decision"] = [
            {"area": "a", "decision": "d", "reference": "NOTES.md", "anchor": "canonical writer"},
            {"area": "a", "decision": "d", "reference": "NOTES.md", "anchor": "absent text"},
            {"area": "a", "decision": "d", "reference": "MISSING.md", "anchor": "x"},
        ]
        errors, _ = snap.validate(doc, self.repo, strict=False)
        self.assertEqual(len(errors), 2)

    def test_base_must_descend_from_tag(self) -> None:
        code, err = run(self.repo, "--write", "--tag", "v1.0.0", "--base", "v0.9.0")
        self.assertEqual(code, 1)
        self.assertIn("does not descend", err)


if __name__ == "__main__":
    unittest.main()
