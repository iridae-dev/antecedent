#!/usr/bin/env python3
"""Fail when documentation or a Python docstring uses a retired evidence term.

A route is supported (provenance, an executed independent check, and for interval routes a
construction check) or `not_supported`; calibration status only annotates. The retired
vocabulary described support as a licence to be granted, a route as frozen or closed, and
calibration as something that refuses. This gate scans prose, so the words cannot return:

* `docs/**`, `README.md`, `python/README.md`, and the docstrings of `python/antecedent/**`;
* not `docs/release-notes/` or `parity/release_snapshots/`, which record what was said then.

Software-licence lines (naming MIT, Apache, SPDX, LICENSE or the cargo-deny policy) and the
plain noun "license" are not flagged.

Usage:
    python3 scripts/gate_docs_vocabulary.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = ("docs/release-notes/", "parity/release_snapshots/")
TERMS = {
    "licensed/licensing": r"\b(?:un)?licen[sc](?:ed|ing)\b",
    "promoted/promotion": r"\bpromot(?:e|es|ed|ing|ion|ions)\b",
    "allowlisted route": r"\ballowlisted\s+routes?\b",
    "executes_unlicensed": r"\bexecutes_unlicensed\b",
    "cell_not_licensed": r"\bcell_not_licensed\b",
    "frozen/withheld/closed route": r"\b(?:frozen|withheld|closed)\s+routes?\b|\bfrozen\s+intervals?\b",
    "waiver": r"\bwaivers?\b",
    "until a coverage record": r"\buntil\s+a\s+coverage\s+record\b",
    "calibration refuses": r"\bcalibration\s+refuses\b",
}
PATTERNS = {name: re.compile(rx, re.I) for name, rx in TERMS.items()}
SOFTWARE_LICENCE = re.compile(r"\b(?:MIT|Apache|SPDX|LICENSE)\b|deny\.toml|cargo-deny")
PROSE_SUFFIXES = {".md", ".txt", ".rst"}


def prose_files() -> list[Path]:
    files = [
        p
        for p in (ROOT / "docs").rglob("*")
        if p.is_file()
        and p.suffix in PROSE_SUFFIXES
        and not p.relative_to(ROOT).as_posix().startswith(EXCLUDED)
    ]
    files += [ROOT / "README.md", ROOT / "python" / "README.md"]
    return sorted(p for p in files if p.is_file())


def docstring_lines(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            value = body[0].value
            if isinstance(value.value, str):
                for k, line in enumerate(value.value.splitlines()):
                    out.append((value.lineno + k, line))
    return out


def scan(lines: list[tuple[int, str]], rel: str) -> list[str]:
    problems = []
    for number, line in lines:
        if SOFTWARE_LICENCE.search(line):
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(line):
                problems.append(f"{rel}:{number}: retired term ({name}): {line.strip()[:100]}")
    return problems


def main() -> int:
    problems: list[str] = []
    for path in prose_files():
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        problems += scan(list(enumerate(text, 1)), path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / "python" / "antecedent").rglob("*.py")):
        problems += scan(docstring_lines(path), path.relative_to(ROOT).as_posix())
    for p in problems:
        print(p)
    if problems:
        print(f"{len(problems)} retired term(s) in documentation")
        return 1
    print("documentation vocabulary ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
