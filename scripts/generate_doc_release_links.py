#!/usr/bin/env python3
"""Refresh live narrative example and Rust API links from the documentation version.

Historical release notes, evidence-linked excluded sources, and the primary
README are intentionally outside this pass.
"""

import argparse
import re

from docs_hooks import publication_exclusions
from generate_example_catalog import ROOT, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    errors = []
    release = version()
    excluded = set(publication_exclusions())
    for path in (ROOT / "docs").rglob("*.md"):
        uri = str(path.relative_to(ROOT / "docs"))
        if uri in excluded or uri.startswith("release-notes/") or uri == "examples.md":
            continue
        text = path.read_text(encoding="utf-8")
        expected = re.sub(
            r"(https://github.com/iridae-dev/antecedent/blob/)(?:main|v[0-9.]+)(/examples/)",
            lambda m: m[1] + "v" + release + m[2],
            text,
        )
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
