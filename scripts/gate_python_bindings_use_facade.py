#!/usr/bin/env python3
"""Tier 0: the Python bindings consume the public `antecedent` facade only.

Fails when a Rust source under `python/src` names an internal crate path
(`antecedent_<crate>`) outside comments and string literals, or when `python/Cargo.toml`
depends on, or forwards a feature to, an internal crate (any `antecedent-<crate>`; the
facade `antecedent` is the one allowed).

Usage: gate_python_bindings_use_facade.py [--root REPO]
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
INTERNAL_IDENT = re.compile(r"\bantecedent_[A-Za-z0-9_]+(?=\s*::)|(?<=extern crate )antecedent_[A-Za-z0-9_]+")
INTERNAL_DEP = re.compile(r"^(?:dep:)?antecedent-[A-Za-z0-9_-]+")
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def code_only(src: str) -> str:
    """Blank comments, string literals and char literals, keeping line structure."""
    out = list(src)
    n = len(src)
    i = 0

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif src[i] == "r" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")) and re.match(r'r#*"', src[i:i + 40]):
            m = re.match(r'r(#*)"', src[i:])
            end = src.find('"' + m.group(1), i + m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            blank(i, j)
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            blank(i, j)
            i = j
        elif src[i] == "'":
            m = re.match(r"'(\\[^']*|[^\\'])'", src[i:i + 12])
            if m:
                blank(i, i + m.end())
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def check_source(path: Path, rel: str) -> list[str]:
    errors = []
    code = code_only(path.read_text(encoding="utf-8"))
    for lineno, line in enumerate(code.splitlines(), 1):
        for m in INTERNAL_IDENT.finditer(line):
            errors.append(f"{rel}:{lineno}: internal crate path `{m.group(0)}`; use the `antecedent` facade")
    return errors


def check_manifest(path: Path, rel: str) -> list[str]:
    errors = []
    manifest = tomllib.loads(path.read_text(encoding="utf-8"))

    def tables(doc):
        for t in DEP_TABLES:
            yield t, doc.get(t, {})
        for tname, target in doc.get("target", {}).items():
            for t in DEP_TABLES:
                yield f"target.{tname}.{t}", target.get(t, {})

    for label, deps in tables(manifest):
        for name in deps:
            pkg = deps[name].get("package", name) if isinstance(deps[name], dict) else name
            for candidate in (name, pkg):
                if INTERNAL_DEP.match(candidate) and candidate != "antecedent":
                    errors.append(f"{rel}: [{label}] depends on internal crate `{candidate}`; depend on `antecedent` only")
                    break
    for feature, members in manifest.get("features", {}).items():
        for member in members:
            if INTERNAL_DEP.match(member):
                errors.append(f"{rel}: feature `{feature}` forwards to internal crate via `{member}`; forward through `antecedent/...`")
    return errors


def run(root: Path) -> list[str]:
    errors = []
    manifest = root / "python" / "Cargo.toml"
    if not manifest.is_file():
        return [f"{manifest}: missing"]
    errors += check_manifest(manifest, "python/Cargo.toml")
    for path in sorted((root / "python" / "src").rglob("*.rs")):
        errors += check_source(path, path.relative_to(root).as_posix())
    return errors


def main(argv: list[str]) -> int:
    root = DEFAULT_ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1])
    errors = run(root)
    for e in errors:
        print(e, file=sys.stderr)
    if errors:
        print(f"gate_python_bindings_use_facade: {len(errors)} violation(s)", file=sys.stderr)
        return 1
    print("gate_python_bindings_use_facade: python bindings use the facade only")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
