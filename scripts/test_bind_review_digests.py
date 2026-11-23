#!/usr/bin/env python3
"""Tests for binding reviewed artifact digests in both TOML layouts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bind_review_digests as bind
import evidence_kinds
from bind_review_digests import bind_artifact_digest


class BindArtifactDigest(unittest.TestCase):
    def test_multiline_and_inline_preserve_surrounding_text(self):
        old, new = "a" * 64, "b" * 64
        for separator in ("\n", ", "):
            text = f'path = "src/example.rs"{separator}sha256 = "{old}"'
            self.assertEqual(
                bind_artifact_digest(text, "src/example.rs", new),
                f'path = "src/example.rs"{separator}sha256 = "{new}"',
            )

    def test_items_line_between_path_and_digest_is_kept(self):
        old, new = "a" * 64, "b" * 64
        for separator in ("\n", ", "):
            text = f'path = "src/example.rs"{separator}items = ["fn a", "impl B"]{separator}sha256 = "{old}"'
            self.assertEqual(
                bind_artifact_digest(text, "src/example.rs", new),
                f'path = "src/example.rs"{separator}items = ["fn a", "impl B"]{separator}sha256 = "{new}"',
            )

    def test_missing_or_duplicate_artifacts_fail_closed(self):
        entry = 'path = "src/example.rs", sha256 = "' + "a" * 64 + '"'
        for text in ("", entry + "\n" + entry):
            with self.assertRaises(SystemExit):
                bind_artifact_digest(text, "src/example.rs", "b" * 64)


class FixtureReviews(unittest.TestCase):
    def test_a_rebind_moves_the_review_out_of_the_fixture_tests_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = root / "conformance/family/case/expected.json"
            fixture.parent.mkdir(parents=True)
            (root / "conformance/family/case/reference.py").write_text("print(1)\n", encoding="utf-8")
            review = {"artifacts": [{"path": "conformance/family/case/reference.py", "role": "generator",
                                     "sha256": "0" * 64}], "fixture_payload_sha256": "0" * 64}
            data = {"case": {"x": 1}, "oracle": {"kind": "enumeration", "reference_review": review}}
            fixture.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            with patch.object(bind, "REPO_ROOT", root):
                bind.bind_fixture(fixture)
            stored = json.loads(fixture.read_text(encoding="utf-8"))
            self.assertNotIn("reference_review", stored["oracle"])
            self.assertEqual(stored["case"], {"x": 1})
            moved = json.loads((root / "parity/fixture_reviews/family/case.json").read_text(encoding="utf-8"))
            self.assertEqual(moved["fixture_payload_sha256"], evidence_kinds.fixture_payload_digest(stored))
            self.assertNotEqual(moved["artifacts"][0]["sha256"], "0" * 64)
            loaded = evidence_kinds.load_fixture(root, fixture)
            self.assertEqual(loaded["oracle"]["reference_review"], moved)
            # Rebinding again leaves the fixture's bytes alone.
            before = fixture.read_bytes()
            with patch.object(bind, "REPO_ROOT", root):
                bind.bind_fixture(fixture)
            self.assertEqual(fixture.read_bytes(), before)

    def test_a_review_kept_in_both_places_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = root / "conformance/family/case/expected.json"
            fixture.parent.mkdir(parents=True)
            fixture.write_text(json.dumps({"oracle": {"reference_review": {}}}), encoding="utf-8")
            (root / "parity/fixture_reviews/family").mkdir(parents=True)
            (root / "parity/fixture_reviews/family/case.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both embedded"):
                evidence_kinds.load_fixture(root, fixture)


if __name__ == "__main__":
    unittest.main()
