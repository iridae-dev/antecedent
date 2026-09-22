#!/usr/bin/env python3
"""Published-docs link check: relative links and repository blob links must resolve.

Two kinds of link are checked in the user-facing Markdown:

* a relative link (`](path)`) must name a file that exists next to the page;
* a GitHub `blob/<ref>/<path>` link into this repository must name a file that
  exists in the working tree when `<ref>` is the release being documented
  (`v<target_version>` from `docs/release-notes/preparation.toml`, else the
  workspace version) or `main`. Links pinned to any other ref are a frozen
  reference to a shipped snapshot and are not checked.

Audit ledgers and release-note history are not scanned: they cite historical
paths on purpose. The roadmap is scanned like any user-facing page.

Run directly, or via scripts/gate_release.sh.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAGES = (
    ["README.md", "CONTRIBUTING.md", "SECURITY.md", "ROADMAP.md", "examples/README.md"]
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md"))
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "adr").glob("*.md"))
)
SKIPPED_PREFIXES = ("docs/audits/", "docs/release-notes/")

LINK = re.compile(r"\]\(([^)\s]+)\)")
REPO_BLOB = re.compile(r"^https://github\.com/iridae-dev/antecedent/blob/([^/]+)/(.+)$")


def documented_ref() -> str:
    prep = ROOT / "docs/release-notes/preparation.toml"
    if prep.is_file():
        target = tomllib.loads(prep.read_text()).get("target_version")
        if isinstance(target, str):
            return f"v{target}"
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    return "v" + cargo["workspace"]["package"]["version"]


def main() -> int:
    live_refs = {documented_ref(), "main"}
    fail: list[str] = []
    for rel in PAGES:
        if rel.startswith(SKIPPED_PREFIXES):
            continue
        page = ROOT / rel
        if not page.is_file():
            continue
        in_fence = False
        for number, line in enumerate(page.read_text().splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            for target in LINK.findall(line):
                blob = REPO_BLOB.match(target)
                if blob:
                    ref, path = blob.group(1), blob.group(2).split("#", 1)[0]
                    if ref in live_refs and not (ROOT / path).exists():
                        fail.append(f"{rel}:{number}: blob/{ref}/{path} does not exist")
                    continue
                if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                if path and not (page.parent / path).exists():
                    fail.append(f"{rel}:{number}: relative link {path} does not exist")
    if fail:
        print("Doc link check FAILED:")
        for item in fail:
            print(f" - {item}")
        return 1
    print(f"Doc link check OK (blob refs checked: {', '.join(sorted(live_refs))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
