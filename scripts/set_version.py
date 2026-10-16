#!/usr/bin/env python3
"""Sync the workspace version and the generated release snippets.

CLI: python3 scripts/set_version.py X.Y.Z 0|1

The second argument is 1 for --check (write nothing). Supported lines and the
reviewed package version live in parity/release.toml. Markdown and notebook
pins are generated blocks, not facts read back out of prose.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(".")

SUPPORTED_BEGIN = "<!-- generated:release:supported-lines:begin -->"
SUPPORTED_END = "<!-- generated:release:supported-lines:end -->"
REVIEWED_BEGIN = "<!-- generated:release:reviewed-version:begin -->"
REVIEWED_END = "<!-- generated:release:reviewed-version:end -->"
CRATE_BEGIN = "<!-- generated:release:crate-dep:begin -->"
CRATE_END = "<!-- generated:release:crate-dep:end -->"
PIN_BEGIN = "<!-- generated:release:python-pin:begin -->"
PIN_END = "<!-- generated:release:python-pin:end -->"

_PIN = re.compile(r"antecedent>=\d+\.\d+\.\d+(?:[.-][0-9A-Za-z.-]+)?,<\d+\.\d+")
_MINOR_CHECK = re.compile(r"!= \(\d+, \d+\)")


class SnippetError(Exception):
    pass


def major_minor(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise SnippetError(f"not a major.minor version: {version}")
    return f"{int(parts[0])}.{int(parts[1])}"


def python_pin(version: str) -> str:
    major, minor = major_minor(version).split(".")
    return f"antecedent>={version},<{major}.{int(minor) + 1}"


def supported_table(lines: list[str]) -> str:
    rows = ["| Version | Supported |", "|---------|-----------|"]
    for line in lines:
        rows.append(f"| {line}.x   | Yes |")
    rows.append("| < 2.0   | No — upgrade to the latest release |")
    return "\n".join(rows)


def reviewed_scope(version: str) -> str:
    return f"Scope: workspace crates + `python` extension (package version **{version}**)"


def crate_dep(version: str) -> str:
    return "\n".join(
        [
            "```toml",
            "[dependencies]",
            f'antecedent = "{major_minor(version)}"',
            "```",
        ]
    )


def examples_install(version: str) -> str:
    return "\n".join(
        [
            "```bash",
            f'python -m pip install "{python_pin(version)}" pandas matplotlib ipykernel jupyterlab',
            "```",
        ]
    )


def replace_generated(text: str, begin: str, end: str, body: str, label: str) -> str:
    pattern = re.compile(re.escape(begin) + r"\n.*?\n" + re.escape(end), re.S)
    replacement = f"{begin}\n{body}\n{end}"
    new, n = pattern.subn(lambda _match: replacement, text, count=1)
    if n != 1:
        raise SnippetError(f"{label}: missing generated block {begin}")
    return new


def rewrite_assignment(text: str, key: str, literal: str) -> str:
    pattern = re.compile(rf"(?m)^({re.escape(key)}\s*=\s*).*$")
    new, n = pattern.subn(rf"\g<1>{literal}", text, count=1)
    if n != 1:
        raise SnippetError(f"parity/release.toml: {key} missing")
    return new


def load_release(root: Path) -> tuple[str, list[str], str | None]:
    path = root / "parity" / "release.toml"
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    lines = data.get("supported_lines")
    if not isinstance(lines, list) or not lines or not all(isinstance(item, str) for item in lines):
        raise SnippetError("parity/release.toml: supported_lines must be a non-empty string array")
    for line in lines:
        if major_minor(f"{line}.0") != line:
            raise SnippetError(f"parity/release.toml: supported line {line!r} is not major.minor")
    reviewed = data.get("reviewed_version")
    if reviewed is not None and not isinstance(reviewed, str):
        raise SnippetError("parity/release.toml: reviewed_version must be a string")
    return text, lines, reviewed


def release_lines_for(version: str, lines: list[str]) -> list[str]:
    minor = major_minor(version)
    if minor in lines:
        return list(lines)
    return [minor, *lines]


def apply_release_snippets(root: Path, version: str, edits: dict[Path, str]) -> None:
    """Fill `edits` with generated release snippets for `version`.

    `supported_lines` gains the new minor when it is absent. `reviewed_version`
    becomes `version`. Callers compare or write `edits`; this function does not
    touch the tree.
    """
    release_path = root / "parity" / "release.toml"
    text, lines, _reviewed = load_release(root)
    new_lines = release_lines_for(version, lines)
    literal = "[" + ", ".join(f'"{line}"' for line in new_lines) + "]"
    text = rewrite_assignment(text, "supported_lines", literal)
    text = rewrite_assignment(text, "reviewed_version", f'"{version}"')
    edits[release_path] = text

    security = root / "SECURITY.md"
    edits[security] = replace_generated(
        security.read_text(encoding="utf-8"),
        SUPPORTED_BEGIN,
        SUPPORTED_END,
        supported_table(new_lines),
        "SECURITY.md",
    )

    review = root / "docs" / "security_review.md"
    edits[review] = replace_generated(
        review.read_text(encoding="utf-8"),
        REVIEWED_BEGIN,
        REVIEWED_END,
        reviewed_scope(version),
        "docs/security_review.md",
    )

    crate = root / "crates" / "antecedent" / "README.md"
    edits[crate] = replace_generated(
        crate.read_text(encoding="utf-8"),
        CRATE_BEGIN,
        CRATE_END,
        crate_dep(version),
        "crates/antecedent/README.md",
    )

    examples = root / "examples" / "README.md"
    edits[examples] = replace_generated(
        examples.read_text(encoding="utf-8"),
        PIN_BEGIN,
        PIN_END,
        examples_install(version),
        "examples/README.md",
    )

    notebooks = sorted((root / "examples" / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 6:
        raise SnippetError(f"expected 6 notebooks, found {len(notebooks)}")
    pin = python_pin(version)
    major, minor = major_minor(version).split(".")
    minor_check = f"!= ({int(major)}, {int(minor)})"
    for path in notebooks:
        notebook = path.read_text(encoding="utf-8")
        rewritten, pins = _PIN.subn(pin, notebook)
        if pins < 1:
            raise SnippetError(f"{path}: no antecedent pin")
        rewritten, checks = _MINOR_CHECK.subn(minor_check, rewritten)
        if checks != 1:
            raise SnippetError(f"{path}: expected one minor-version check, found {checks}")
        edits[path] = rewritten


def release_snippet_problems(root: Path, version: str) -> list[str]:
    """Files whose generated release snippets are not already at `version`."""
    edits: dict[Path, str] = {}
    try:
        apply_release_snippets(root, version, edits)
    except SnippetError as exc:
        return [str(exc)]
    stale = [str(path) for path, new in edits.items() if path.read_text(encoding="utf-8") != new]
    if not stale:
        return []
    return [f"release snippets are not at version {version}: " + ", ".join(stale)]


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: set_version.py X.Y.Z 0|1")
    version = argv[1]
    check = argv[2] == "1"
    root = ROOT
    script = root / "scripts" / "generate_support_matrix_docs.py"

    # path -> new text; nothing is written until every entry has been computed.
    edits: dict[Path, str] = {}

    cargo = root / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    matched = re.search(r"(?ms)^\[workspace\.package\]\n(.*?)(?=\n\[|\Z)", text)
    if not matched:
        sys.exit("Cargo.toml: [workspace.package] not found")
    old_m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', matched.group(0))
    if not old_m:
        sys.exit("Cargo.toml: version missing under [workspace.package]")
    old_version = old_m.group(1)
    block = matched.group(0)
    block_new, n = re.subn(
        r'(?m)^(version\s*=\s*")[^"]*(")',
        rf"\g<1>{version}\2",
        block,
        count=1,
    )
    if n != 1:
        sys.exit("Cargo.toml: workspace.package version not updated")
    edits[cargo] = text[: matched.start()] + block_new + text[matched.end() :]

    pyproject = root / "python" / "pyproject.toml"
    py_new, n = re.subn(
        r'(?m)^(version\s*=\s*")[^"]*(")',
        rf"\g<1>{version}\2",
        pyproject.read_text(encoding="utf-8"),
        count=1,
    )
    if n != 1:
        sys.exit("python/pyproject.toml: version not updated")
    edits[pyproject] = py_new

    # Path-dep version pins must match for crates.io packaging.
    path_pat = re.compile(
        r'(antecedent(?:-[a-z0-9-]+)?\s*=\s*\{\s*version\s*=\s*")[^"]+(")'
    )
    for path in sorted(root.glob("crates/*/Cargo.toml")) + [root / "python" / "Cargo.toml"]:
        if not path.is_file():
            continue
        current = path.read_text(encoding="utf-8")
        updated, n = path_pat.subn(rf"\g<1>{version}\2", current)
        if n:
            edits[path] = updated

    init = root / "python" / "antecedent" / "__init__.py"
    init_text = init.read_text(encoding="utf-8")
    # A semver literal is kept in lockstep. Otherwise __version__ comes from the
    # extension or installed metadata; "unknown" is the sentinel when neither is
    # available, not a version pin.
    literal = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    if literal and re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?", literal.group(1)
    ):
        init_new, n = re.subn(
            r'(__version__\s*=\s*")[^"]+(")',
            rf"\g<1>{version}\2",
            init_text,
            count=1,
        )
        if n != 1:
            sys.exit("python/antecedent/__init__.py: fallback __version__ not updated")
        edits[init] = init_new
    elif "from ._native import __version__ as __version__" not in init_text:
        sys.exit("python/antecedent/__init__.py: fallback __version__ not updated")

    uv = root / "python" / "uv.lock"
    if uv.is_file():
        uv_new, n = re.subn(
            r'(name = "antecedent"\nversion = ")[^"]+(")',
            rf"\g<1>{version}\2",
            uv.read_text(encoding="utf-8"),
            count=1,
        )
        if n == 1:
            edits[uv] = uv_new

    cff = root / "CITATION.cff"
    if cff.is_file():
        cff_new, n = re.subn(
            r"(?m)^(version:\s*)\S+",
            rf"\g<1>{version}",
            cff.read_text(encoding="utf-8"),
            count=1,
        )
        if n != 1:
            sys.exit("CITATION.cff: version not updated")
        edits[cff] = cff_new

    try:
        apply_release_snippets(root, version, edits)
    except SnippetError as exc:
        sys.exit(str(exc))

    changed = [path for path, new in edits.items() if path.read_text(encoding="utf-8") != new]

    if check:
        lock = root / "Cargo.lock"
        lock_ok = lock.is_file() and re.search(
            rf'name = "antecedent"\nversion = "{re.escape(version)}"', lock.read_text(encoding="utf-8")
        )
        stale = [str(path) for path in changed] + ([] if lock_ok else ["Cargo.lock"])
        if stale:
            print(f"tree is not at version {version}; stale files:", file=sys.stderr)
            for path in stale:
                print(f"  {path}", file=sys.stderr)
            sys.exit(1)
        print(f"tree is at version {version}")
        sys.exit(0)

    for path in changed:
        path.write_text(edits[path], encoding="utf-8", newline="\n")

    if old_version != version:
        subprocess.check_call([sys.executable, str(script), "--freeze", old_version])

    print(f"set version to {version}")


if __name__ == "__main__":
    main(sys.argv)
