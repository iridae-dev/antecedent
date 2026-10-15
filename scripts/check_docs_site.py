#!/usr/bin/env python3
"""Check the actual built narrative/API HTML and its local navigation targets."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from docs_hooks import publication_exclusions
from generate_example_catalog import entries


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])


def check(site):
    pages = {path: Page(path.read_text()) for path in site.rglob("*.html")}
    errors = []
    for path, page in pages.items():
        if "{{bound:" in path.read_text():
            errors.append(f"{path.relative_to(site)}: unresolved bound")
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            if not url.path:
                target = path
            else:
                target = (
                    site / unquote(url.path).lstrip("/")
                    if url.path.startswith("/")
                    else path.parent / unquote(url.path)
                ).resolve()
                if target.is_dir():
                    target /= "index.html"
            if not target.is_relative_to(site):
                errors.append(f"{path.relative_to(site)}: link escapes site: {href}")
            elif not target.is_file():
                errors.append(f"{path.relative_to(site)}: missing target {href}")
            elif (
                url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(site)}: missing anchor {href}")
    for excluded in publication_exclusions():
        html = site / Path(excluded).with_suffix("") / "index.html"
        if html.exists():
            errors.append(f"Excluded source shipped: {excluded}")
    for row in entries():
        if "notebook" in row:
            preview = site / "notebook-previews" / (Path(row["notebook"]).stem + ".html")
            if not preview.is_file():
                errors.append(f"Missing notebook preview: {row['id']}")
    required = [
        "index.html",
        "support-lookup/index.html",
        "python/antecedent.html",
        "python/antecedent/measurement_models.html",
        "python/antecedent/evidence_ledger.html",
    ]
    for relative in required:
        if not (site / relative).is_file():
            errors.append(f"Missing front-door page: {relative}")
    return errors, len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    errors, count = check(args.site.resolve())
    if errors:
        print("Built documentation FAILED:")
        for error in errors:
            print(" -", error)
        return 1
    print(
        f"Built documentation verified: {count} HTML pages, local links/anchors, exclusions, notebook previews and public API entry points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
