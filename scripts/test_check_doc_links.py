"""The published-docs link rules, run with the script unit tests."""

from __future__ import annotations

import contextlib
import io
import unittest

from _repo import REPO_ROOT, documented_ref, ensure_scripts_on_path, source_url

ensure_scripts_on_path()
import check_doc_links  # noqa: E402


class DocLinkRulesTest(unittest.TestCase):
    def test_link_rules_hold_on_fixture_pages(self) -> None:
        # Covers anchors and the refusal of a published page that links repository
        # sources at `main` or at another release tag.
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = check_doc_links.self_test()
        self.assertEqual(status, 0, output.getvalue())

    def test_generated_source_links_name_the_documented_release(self) -> None:
        ref = documented_ref(REPO_ROOT)
        self.assertRegex(ref, r"^v\d+\.\d+\.\d+$")
        self.assertEqual(
            source_url("parity/README.md"),
            f"https://github.com/iridae-dev/antecedent/blob/{ref}/parity/README.md",
        )


if __name__ == "__main__":
    unittest.main()
