#!/usr/bin/env python3
"""Generate example notebooks from their Python scripts.

Every catalog row with a ``notebook`` also names a ``python`` script; the script is the
single source and the notebook is generated from it. Scripts mark cells in percent format:

    # %% [markdown]
    # ## A heading
    #
    # Prose, one ``# `` comment line per Markdown line.

    # %%
    code()

Code before the first marker becomes the first code cell. The module docstring becomes
the notebook introduction under the catalog title, followed by the release-pinned Colab
badge and the standard setup cells. ``from __future__`` imports are dropped (a notebook
cell is compiled on its own). Scripts run top to bottom, so no ``__main__`` guard.

    python scripts/generate_notebooks.py          # write
    python scripts/generate_notebooks.py --check  # fail on drift
"""

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

from generate_example_catalog import entries, version

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")
METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.0",
    },
}
SETUP = """## Setup

The next cell installs Antecedent from PyPI when it is not already present (Google Colab).
Use **Runtime → Run all**. If you upgrade after importing it, restart the kernel before
continuing."""


def pin(release):
    major, minor = release.split(".")[:2]
    return f"antecedent>={release},<{major}.{int(minor) + 1}", (int(major), int(minor))


def install_cell(release, packages):
    requirement, minor = pin(release)
    listed = "".join(f'            "{name}",\n' for name in [requirement, *packages])
    return f"""import importlib.util
import subprocess
import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "Antecedent needs Python 3.11 or later. In Colab: Runtime → Change runtime type."
    )
if importlib.util.find_spec("antecedent") is None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
{listed}        ]
    )

import antecedent

if tuple(int(part) for part in antecedent.__version__.split(".")[:2]) != {minor}:
    raise RuntimeError(
        "This notebook requires a different Antecedent minor. "
        "Upgrade with pip install -U '{requirement}' and restart the kernel."
    )"""


def split_script(source, path):
    """Return (docstring, [(kind, text)]) from a percent-format script."""
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree, clean=True) or ""
    lines = source.splitlines()
    start = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and docstring:
        start = tree.body[0].end_lineno
    elif lines and lines[0].startswith("#!"):
        start = 1
    cells, kind, current = [], "code", []

    def flush():
        text = "\n".join(current).strip("\n")
        if kind == "markdown":
            body = []
            for line in text.splitlines():
                if line != "#" and not line.startswith("# "):
                    raise SystemExit(f"{path}: markdown cell line is not a '# ' comment: {line!r}")
                body.append(line[2:])
            text = "\n".join(body).strip("\n")
        if text:
            cells.append((kind, text))

    for line in lines[start:]:
        match = MARKER.match(line)
        if match:
            flush()
            kind, current = ("markdown" if match.group(1) else "code"), []
        elif line.startswith("from __future__ import"):
            continue
        elif line.startswith("if __name__ =="):
            raise SystemExit(f"{path}: notebook scripts run top to bottom; drop the __main__ guard")
        else:
            current.append(line)
    flush()
    return docstring, cells


def markdown_text(text):
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def cell(kind, text, notebook, index):
    digest = hashlib.sha256(f"{notebook}:{index}".encode()).hexdigest()[:12]
    made = {
        "cell_type": kind,
        "id": digest,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }
    if kind == "code":
        made |= {"execution_count": None, "outputs": []}
    return made


def render(row, release):
    script = ROOT / row["python"]
    notebook = row["notebook"]
    docstring, body = split_script(script.read_text(encoding="utf-8"), row["python"])
    if not any(kind == "markdown" for kind, _ in body):
        raise SystemExit(f"{row['python']}: no '# %% [markdown]' cells; mark the script's sections")
    colab = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/iridae-dev/antecedent/blob/v{release}/{notebook})"
    )
    intro = "\n\n".join(
        part
        for part in [
            f"# {row['title']}",
            markdown_text(docstring),
            colab,
            "**Repository:** <https://github.com/iridae-dev/antecedent>  \n"
            "**Documentation:** <https://antecedent.readthedocs.io/>  \n"
            f"**Script:** [`{row['python']}`](https://github.com/iridae-dev/antecedent/blob/v{release}/{row['python']})",
        ]
        if part
    )
    packages = ["numpy"] + [
        name
        for name in row.get("dependencies", [])
        if "(optional)" not in name and name not in ("numpy", "ipykernel")
    ]
    sections = [
        ("markdown", intro),
        ("markdown", SETUP),
        ("code", install_cell(release, packages)),
        *body,
    ]
    document = {
        "cells": [cell(kind, text, notebook, index) for index, (kind, text) in enumerate(sections)],
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def expected():
    release = version()
    rows = [row for row in entries() if "notebook" in row]
    missing = [row["id"] for row in rows if "python" not in row]
    if missing:
        raise SystemExit(f"Notebook rows without a source script: {missing}")
    return {ROOT / row["notebook"]: render(row, release) for row in rows}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail when a notebook differs")
    args = parser.parse_args()
    stale = []
    for path, text in expected().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            if args.check:
                stale.append(str(path.relative_to(ROOT)))
            else:
                path.write_text(text, encoding="utf-8")
                print(f"Wrote {path.relative_to(ROOT)}")
    if stale:
        print("Notebooks differ from their scripts (run python scripts/generate_notebooks.py):")
        for name in stale:
            print(" -", name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
