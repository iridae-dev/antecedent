#!/usr/bin/env python3
"""Refresh live repository-source and Rust API links from the documentation version.

Every published page links repository sources (evidence records, tests,
examples, ADRs) at the documented release (`_repo.documented_ref`), so a
released page keeps naming the files it shipped with. This pass is the one
place that pins them: it rewrites a `main` or release-tag link to that ref, and
`scripts/check_doc_links.py` fails a published page that still names another.
The tag exists only once the release commit is tagged; until then the links
name nothing on GitHub and are checked against the working tree.

Historical release notes, evidence-linked excluded sources, and the primary
README are intentionally outside this pass.
"""

import argparse
import re

from _repo import REPOSITORY_URL, documented_ref
from docs_hooks import publication_exclusions
from generate_example_catalog import ROOT, version

SOURCE_LINK = re.compile(
    "(" + re.escape(REPOSITORY_URL) + r"/(?:blob|tree)/)(?:main|v\d+\.\d+\.\d+)(?=/)"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    errors = []
    release = version()
    ref = documented_ref(ROOT)
    excluded = set(publication_exclusions())
    for path in (ROOT / "docs").rglob("*.md"):
        uri = str(path.relative_to(ROOT / "docs"))
        if uri in excluded or uri.startswith("release-notes/") or uri == "examples.md":
            continue
        text = path.read_text(encoding="utf-8")
        expected = SOURCE_LINK.sub(lambda m: m[1] + ref, text)
        expected = re.sub(
            r"https://docs\.rs/antecedent(?:/[0-9.]+/antecedent/)?(?=[)\s])",
            "https://docs.rs/antecedent/" + release + "/antecedent/",
            expected,
        )
        if uri == "index.md":
            expected = re.sub(
                r"(\[release notes\]\(release-notes/)v[0-9.]+\.md",
                lambda m: m[1] + "v" + release + ".md",
                expected,
            )
        if expected != text:
            if args.check:
                errors.append(uri)
            else:
                path.write_text(expected, encoding="utf-8")
    for path in (ROOT / "examples/notebooks").glob("*.ipynb"):
        text = path.read_text(encoding="utf-8")
        expected = re.sub(
            r"(https://(?:colab\.research\.google\.com/github|github\.com)/iridae-dev/antecedent/blob/)(?:main|v[0-9.]+)(/examples/)",
            lambda match: match[1] + "v" + release + match[2],
            text,
        )
        if expected != text:
            if args.check:
                errors.append(str(path.relative_to(ROOT)))
            else:
                path.write_text(expected, encoding="utf-8")
    if errors:
        raise SystemExit("Stale current-release links: " + ", ".join(errors))
    print("Current-release documentation links verified")


if __name__ == "__main__":
    main()
