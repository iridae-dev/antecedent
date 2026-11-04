#!/usr/bin/env python3
"""Tests for binding reviewed artifact digests in both TOML layouts."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
