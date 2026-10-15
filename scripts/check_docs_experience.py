#!/usr/bin/env python3
"""Check public docs boundaries, example coverage, embeds and release consistency."""

import ast
import re
import sys
from pathlib import Path

from check_doc_links import PAGES, check_pages
from docs_hooks import (
    EXAMPLE,
    publication_exclusions,
    render_bounds,
    render_examples,
    split_release,
)
from generate_example_catalog import entries, expected_page, version

ROOT = Path(__file__).resolve().parents[1]


def check():
    errors = check_pages(ROOT, list(PAGES), anchors=True)
    excluded = set(publication_exclusions())
    current = "v" + version()
    for path in (ROOT / "docs").rglob("*.md"):
        uri = str(path.relative_to(ROOT / "docs"))
        if uri in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            rendered = render_bounds(render_examples(text))
            ast_examples = [ROOT / match.group(1) for match in EXAMPLE.finditer(text)]
            for example in ast_examples:
                ast.parse(example.read_text(encoding="utf-8"))
            split_release(rendered, uri)
        except (ValueError, SyntaxError) as error:
            errors.append(f"{uri}: {error}")
        if not uri.startswith("release-notes/"):
            for ref in re.findall(r"github.com/iridae-dev/antecedent/blob/([^/]+)/examples/", text):
                if ref != current:
                    errors.append(f"{uri}: example link pins {ref}, expected {current}")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                pathpart = target.split("#")[0]
                if not re.match(r"^[a-z]+:", pathpart) and pathpart:
                    linked = (path.parent / pathpart).resolve()
                    if (
                        linked.is_relative_to(ROOT / "docs")
                        and str(linked.relative_to(ROOT / "docs")) in excluded
                    ):
                        errors.append(f"{uri}: links excluded publication source {pathpart}")
    for notebook in (ROOT / "examples/notebooks").glob("*.ipynb"):
        for ref in re.findall(
            r"(?:github\.com|colab\.research\.google\.com/github)/iridae-dev/antecedent/blob/([^/]+)/examples/",
            notebook.read_text(encoding="utf-8"),
        ):
            if ref != current:
                errors.append(f"{notebook.name}: notebook link pins {ref}, expected {current}")
    rows = entries()
    declared = [row[key] for row in rows for key in ("python", "rust", "notebook") if key in row]
    if len(declared) != len(set(declared)):
        errors.append("Duplicate example paths in catalog")
    shipped = {
        str(p.relative_to(ROOT))
        for folder, pattern in [
            ("python", "*.py"),
            ("rust", "*.rs"),
            ("notebooks", "*.ipynb"),
        ]
        for p in (ROOT / "examples" / folder).glob(pattern)
    }
    if set(declared) != shipped:
        errors.append(
            f"Example catalog differs from shipped examples: {sorted(set(declared) ^ shipped)}"
        )
    for row in rows:
        if "**" in row.get("title", "") or re.match(
            r"^(beginner|intermediate|advanced)\s*/", row.get("summary", "")
        ):
            errors.append(
                f"{row['id']}: catalog contains generated table formatting instead of a curated title/summary"
            )
        for key in ("title", "summary", "topic", "level", "runtime", "claim"):
            if not row.get(key):
                errors.append(f"{row['id']}: missing catalog {key}")
    for site, relative in [(False, "examples/README.md"), (True, "docs/examples.md")]:
        if (ROOT / relative).read_text(encoding="utf-8") != expected_page(site):
            errors.append(f"{relative}: generated catalog stale")
    return errors


def main():
    errors = check()
    if errors:
        print("Documentation experience check FAILED:")
        for error in errors:
            print(" -", error)
        return 1
    print(
        "Documentation experience check passed (publication boundaries, release links, embedded sources, complete catalogs, anchors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
