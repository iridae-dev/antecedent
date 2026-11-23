#!/usr/bin/env python3
"""Render the GitHub Release body from the curated release notes.

The notes file carries the readable highlights and then the generated claim
inventory, which is far longer than a Release body may be. The documentation
site already publishes the two as separate pages (`split_release` in
docs_hooks.py); the Release body is the same highlights, followed by a link to
the inventory page of that release's documentation.

    python3 scripts/release_body.py docs/release-notes/<tag>.md \\
        --tag <tag> --repository <owner>/<name> > RELEASE_BODY.md
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

from docs_hooks import ROOT, claims_page_uri, split_claims

NOTES_DIR = "docs/release-notes"
# GitHub refuses a Release body longer than this many characters.
BODY_LIMIT = 125_000
LINK = re.compile(r"\]\(([^)\s]+)\)")


def site_url(root: Path = ROOT) -> str:
    """The documentation site root declared in mkdocs.yml."""
    match = re.search(r"(?m)^site_url:\s*(\S+)\s*$", (root / "mkdocs.yml").read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("mkdocs.yml: site_url not found")
    return match.group(1).rstrip("/")


def claims_page_url(tag: str, site: str) -> str:
    """Where the documentation built from `tag` serves that release's inventory page."""
    page = claims_page_uri(tag).removesuffix(".md")
    return f"{site.rstrip('/')}/en/{tag}/{page}/"


def absolute_links(markdown: str, tag: str, repository: str) -> str:
    """Point the notes' site-relative links at the tagged sources.

    A Release body resolves a relative link against the releases page, where
    `../guides/x.md` names nothing.
    """
    blob = f"https://github.com/{repository}/blob/{tag}/"

    def absolute(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith("#") or re.match(r"^[a-z][a-z0-9+.-]*:", target):
            return match.group(0)
        path, sep, fragment = target.partition("#")
        return f"]({blob}{posixpath.normpath(posixpath.join(NOTES_DIR, path))}{sep}{fragment})"

    return LINK.sub(absolute, markdown)


def release_body(markdown: str, tag: str, repository: str, site: str) -> str:
    highlights, inventory = split_claims(markdown)
    body = absolute_links(highlights, tag, repository).rstrip("\n") + "\n"
    if inventory is not None:
        source = f"https://github.com/{repository}/blob/{tag}/{NOTES_DIR}/{tag}.md"
        body += (
            "\n## Complete claim inventory\n\n"
            "[Per-cell claims, explicit refusals and support inventory]"
            f"({claims_page_url(tag, site)}) are generated from the release evidence. "
            "Read those exact scopes before using a new route. The same inventory is in the "
            f"[tagged release notes]({source}) and in the attached documentation bundle.\n"
        )
    if len(body) > BODY_LIMIT:
        raise SystemExit(
            f"release body is {len(body)} characters; GitHub accepts at most {BODY_LIMIT}"
        )
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notes", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", required=True, help="owner/name")
    parser.add_argument("--site", default=None, help="documentation site root (mkdocs.yml)")
    args = parser.parse_args(argv)
    sys.stdout.write(
        release_body(
            args.notes.read_text(encoding="utf-8"),
            args.tag,
            args.repository,
            args.site or site_url(),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
