#!/usr/bin/env python3
"""Published-docs link check: relative links and repository blob links must resolve.

Two kinds of link are checked in the user-facing Markdown:

* a relative link (`](path)`) must name a file that exists next to the page;
* a GitHub `blob/<ref>/<path>` link into this repository must name a file that
  exists in the working tree when `<ref>` is the release being documented
  (`v<target_version>` from optional `docs/release-notes/preparation.toml`, else
  the workspace version; a missing file is that fallback, not an error) or
  `main`. Links pinned to any other ref are a frozen
  reference to a shipped snapshot and are not checked.

`--anchors` also resolves a `#fragment` against the target page's headings
(GitHub's slug: lowercase, punctuation dropped, spaces to hyphens, duplicates
suffixed `-1`, `-2`, ...). The fragment check is optional so a heading reword
is a link failure only when the caller asks for it. Release notes are not
scanned.

Run directly, or via scripts/gate_release.sh.

    python3 scripts/check_doc_links.py
    python3 scripts/check_doc_links.py --anchors
    python3 scripts/check_doc_links.py --self-test
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as ROOT

PAGES = (
    ["README.md", "CONTRIBUTING.md", "SECURITY.md", "ROADMAP.md", "examples/README.md"]
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md"))
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "adr").glob("*.md"))
)
SKIPPED_PREFIXES = ("docs/release-notes/",)

LINK = re.compile(r"\]\(([^)\s]+)\)")
REPO_BLOB = re.compile(r"^https://github\.com/iridae-dev/antecedent/blob/([^/]+)/(.+)$")


def heading_slug(heading: str) -> str:
    """GitHub-style anchor for a markdown heading line or its text."""
    text = re.sub(r"^#{1,6}\s*", "", heading).strip()
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    text = text.lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip())
    return text


def heading_anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not re.match(r"^#{1,6}\s+\S", line):
            continue
        slug = heading_slug(line)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def documented_ref(root: Path = ROOT) -> str:
    prep = root / "docs/release-notes/preparation.toml"
    if prep.is_file():
        target = tomllib.loads(prep.read_text(encoding="utf-8")).get("target_version")
        if isinstance(target, str):
            return f"v{target}"
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    return "v" + cargo["workspace"]["package"]["version"]


def check_pages(root: Path, pages: list[str], *, anchors: bool) -> list[str]:
    live_refs = {documented_ref(root), "main"}
    fail: list[str] = []
    cache: dict[Path, set[str]] = {}

    def anchors_of(path: Path) -> set[str]:
        if path not in cache:
            cache[path] = (
                heading_anchors(path.read_text(encoding="utf-8")) if path.is_file() else set()
            )
        return cache[path]

    for rel in pages:
        if rel.startswith(SKIPPED_PREFIXES):
            continue
        page = root / rel
        if not page.is_file():
            continue
        in_fence = False
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            for target in LINK.findall(line):
                blob = REPO_BLOB.match(target)
                if blob:
                    ref, blob_path = blob.group(1), blob.group(2)
                    path, _, frag = blob_path.partition("#")
                    if ref in live_refs and not (root / path).exists():
                        fail.append(f"{rel}:{number}: blob/{ref}/{path} does not exist")
                    elif (
                        anchors
                        and frag
                        and ref in live_refs
                        and frag not in anchors_of(root / path)
                    ):
                        fail.append(
                            f"{rel}:{number}: blob/{ref}/{path}#{frag} does not match a heading"
                        )
                    continue
                if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                    continue
                path, _, frag = target.partition("#")
                virtual_notebook = re.fullmatch(r"notebook-previews/([\w-]+)\.html", path)
                if virtual_notebook and rel == "docs/examples.md":
                    if not (
                        root / "examples/notebooks" / (virtual_notebook.group(1) + ".ipynb")
                    ).is_file():
                        fail.append(f"{rel}:{number}: notebook preview has no source")
                    continue
                if path and not (page.parent / path).exists():
                    fail.append(f"{rel}:{number}: relative link {path} does not exist")
                    continue
                if anchors and frag:
                    dest = page if not path else page.parent / path
                    if frag not in anchors_of(dest):
                        fail.append(
                            f"{rel}:{number}: #{frag} does not match a heading in {dest.name}"
                        )
    return fail


def self_test() -> int:
    import tempfile

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "Cargo.toml").write_text(
            '[workspace.package]\nversion = "9.9.9"\n', encoding="utf-8", newline="\n"
        )
        (root / "docs/page.md").write_text(
            "# Title\n\n## How to read the claims\n\nSee [claims](#how-to-read-the-claims) and [missing](#nope).\n",
            encoding="utf-8",
            newline="\n",
        )
        (root / "docs/other.md").write_text(
            "## Explicit refusals\n", encoding="utf-8", newline="\n"
        )
        pages = ["docs/page.md", "docs/other.md"]
        plain = check_pages(root, pages, anchors=False)
        if plain:
            failures.append(f"fragments are ignored unless --anchors: {plain}")
        (root / "docs/page.md").write_text(
            (root / "docs/page.md").read_text(encoding="utf-8")
            + "\nSee [other](other.md#explicit-refusals).\n",
            encoding="utf-8",
            newline="\n",
        )
        anchored = check_pages(root, pages, anchors=True)
        if not any("#nope" in item for item in anchored):
            failures.append(f"missing anchor was accepted: {anchored}")
        if any(
            "explicit-refusals" in item or "how-to-read-the-claims" in item for item in anchored
        ):
            failures.append(f"live headings were rejected: {anchored}")
        # Rewording a heading breaks only the optional anchor check, and only the link that named it.
        (root / "docs/other.md").write_text(
            "## Refusals, restated\n", encoding="utf-8", newline="\n"
        )
        moved = check_pages(root, pages, anchors=True)
        if not any("explicit-refusals" in item for item in moved):
            failures.append(f"reworded heading did not break its anchor: {moved}")
        if check_pages(root, pages, anchors=False):
            failures.append("reworded heading failed the check with anchors off")
    if failures:
        print("check_doc_links self-test FAILED:")
        for item in failures:
            print(f" - {item}")
        return 1
    print("check_doc_links self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in args:
        return self_test()
    anchors = "--anchors" in args
    fail = check_pages(ROOT, list(PAGES), anchors=anchors)
    if fail:
        print("Doc link check FAILED:")
        for item in fail:
            print(f" - {item}")
        return 1
    suffix = ", anchors" if anchors else ""
    print(
        "Doc link check OK (blob refs checked: "
        f"{', '.join(sorted({documented_ref(), 'main'}))}{suffix})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
