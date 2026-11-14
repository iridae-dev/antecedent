"""The release-notes split shared by the documentation site and the GitHub Release body."""

from __future__ import annotations

import unittest

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import release_body as rb  # noqa: E402
from docs_hooks import claims_page_uri, split_claims, split_release  # noqa: E402

NOTES = """# Release

Read [the guide](../guides/thing.md#part), [a sibling](other.md), [here](#migrate)
and [elsewhere](https://example.org/x).

## Migrate

Nothing.

## Claims by cell

<!-- generated:release-claims:cells:begin -->
| cell | claim |
"""
REPOSITORY = "owner/name"
SITE = "https://docs.example/"


class SplitTest(unittest.TestCase):
    def test_notes_split_at_the_inventory_heading_and_rejoin_exactly(self) -> None:
        highlights, inventory = split_claims(NOTES)
        self.assertEqual(highlights + inventory, NOTES)
        self.assertTrue(inventory.startswith("## Claims by cell"))
        self.assertNotIn("generated:release-claims", highlights)
        self.assertEqual(split_claims("# Notes\n"), ("# Notes\n", None))

    def test_site_pages_are_the_highlights_and_a_linked_inventory_page(self) -> None:
        highlights, (uri, page) = split_release(NOTES, "release-notes/vX.md")
        self.assertEqual(uri, claims_page_uri("vX"))
        self.assertIn(f"(../{uri})", highlights)
        self.assertNotIn("generated:release-claims", highlights)
        self.assertEqual(page, "# vX claim and support inventory\n\n" + split_claims(NOTES)[1])
        # Only release notes are split, and only when they carry an inventory.
        self.assertEqual(split_release(NOTES, "guides/x.md"), (NOTES, None))
        self.assertEqual(split_release("# Notes\n", "release-notes/vX.md"), ("# Notes\n", None))


class ReleaseBodyTest(unittest.TestCase):
    def test_body_is_the_highlights_with_a_link_to_the_published_inventory(self) -> None:
        body = rb.release_body(NOTES, "vX", REPOSITORY, SITE)
        self.assertNotIn("generated:release-claims", body)
        self.assertNotIn("## Claims by cell", body)
        self.assertIn("## Migrate", body)
        self.assertIn("(https://docs.example/en/vX/reference/release-claims-vX/)", body)
        self.assertIn("(https://github.com/owner/name/blob/vX/docs/release-notes/vX.md)", body)

    def test_relative_links_point_at_the_tagged_sources(self) -> None:
        body = rb.release_body(NOTES, "vX", REPOSITORY, SITE)
        blob = "https://github.com/owner/name/blob/vX/docs/"
        self.assertIn(f"[the guide]({blob}guides/thing.md#part)", body)
        self.assertIn(f"[a sibling]({blob}release-notes/other.md)", body)
        self.assertIn("[here](#migrate)", body)
        self.assertIn("[elsewhere](https://example.org/x)", body)

    def test_notes_without_an_inventory_are_published_whole(self) -> None:
        self.assertEqual(rb.release_body("# Notes\n", "vX", REPOSITORY, SITE), "# Notes\n")

    def test_an_oversized_body_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            rb.release_body("x" * (rb.BODY_LIMIT + 1), "vX", REPOSITORY, SITE)

    def test_every_committed_release_notes_file_renders_a_body_github_accepts(self) -> None:
        site = rb.site_url()
        self.assertTrue(site.startswith("https://"))
        notes = sorted((REPO_ROOT / rb.NOTES_DIR).glob("v*.md"))
        self.assertTrue(notes)
        for path in notes:
            with self.subTest(notes=path.name):
                text = path.read_text(encoding="utf-8")
                body = rb.release_body(text, path.stem, REPOSITORY, site)
                self.assertLessEqual(len(body), rb.BODY_LIMIT)
                self.assertNotIn("generated:release-claims", body)
                self.assertNotIn("](../", body)
                if split_claims(text)[1] is not None:
                    self.assertIn(rb.claims_page_url(path.stem, site), body)


if __name__ == "__main__":
    unittest.main()
