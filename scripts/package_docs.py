#!/usr/bin/env python3
"""Package public Markdown with readable bounds and no branch audit sources."""

import argparse
import io
import tarfile
from pathlib import Path

from docs_hooks import (
    ROOT,
    publication_exclusions,
    render_bounds,
    render_examples,
    split_release,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    excluded = set(publication_exclusions())
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    paths += sorted((ROOT / "adr").glob("*.md"))
    paths += sorted((ROOT / "docs/assets").glob("*.svg"))
    paths += [
        p
        for p in sorted((ROOT / "docs").rglob("*.md"))
        if str(p.relative_to(ROOT / "docs")) not in excluded
    ]
    with tarfile.open(args.output, "w:gz") as archive:
        for path in paths:
            markdown = render_bounds(render_examples(path.read_text()))
            appendix = None
            if path.is_relative_to(ROOT / "docs"):
                markdown, appendix = split_release(markdown, str(path.relative_to(ROOT / "docs")))
            data = markdown.encode()
            info = tarfile.TarInfo(str(path.relative_to(ROOT)))
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
            if appendix:
                extra = appendix[1].encode()
                info = tarfile.TarInfo("docs/" + appendix[0])
                info.size = len(extra)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(extra))
    print(f"Public documentation: {len(paths)} files → {args.output}")


if __name__ == "__main__":
    main()
