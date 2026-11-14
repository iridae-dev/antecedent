#!/usr/bin/env python3
"""Execute marked documentation snippets and the exact sources embedded in guides."""

import argparse
import re
import runpy
import sys
import time

from docs_hooks import EXAMPLE, ROOT, publication_exclusions

EXEC = re.compile(r"<!-- exec: ([\w-]+) -->\s*```python\n(.*?)\n```", re.S)


def published_pages():
    """Every published page and its text, in the order the snippets run."""
    excluded = set(publication_exclusions())
    for path in sorted((ROOT / "docs").rglob("*.md")):
        if str(path.relative_to(ROOT / "docs")) not in excluded:
            yield path, path.read_text(encoding="utf-8")


def embedded_sources():
    """The example sources published pages embed, which a full run executes."""
    return sorted(
        {match.group(1) for _, text in published_pages() for match in EXAMPLE.finditer(text)}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snippets-only", action="store_true")
    args = parser.parse_args()
    executed = 0
    for path, text in published_pages():
        contexts = {}
        for context, code in EXEC.findall(text):
            namespace = contexts.setdefault(context, {"__name__": "__main__"})
            exec(compile(code, str(path), "exec"), namespace)
            executed += 1
        if path.name == "first-analysis.md":
            ns = contexts["quickstart"]
            import numpy as np

            assert abs(ns["result"].answer.value - 2) < 0.5
            assert np.isclose(ns["updated"].answer.value, ns["result"].answer.value + 1)
            assert ns["loaded"].answer == ns["result"].answer
    sources = embedded_sources()
    if not args.snippets_only:
        original = sys.argv
        try:
            for source in sources:
                sys.argv = [str(ROOT / source)]
                start = time.monotonic()
                runpy.run_path(str(ROOT / source), run_name="__main__")
                print(
                    f"Embedded example passed: {source} ({time.monotonic() - start:.2f}s)",
                    flush=True,
                )
        finally:
            sys.argv = original
    print(
        f"Executed {executed} published snippets; {len(sources)} distinct embedded example sources"
    )


if __name__ == "__main__":
    main()
