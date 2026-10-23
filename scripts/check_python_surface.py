#!/usr/bin/env python3
"""Rust-to-Python facade parity: every public facade entry names its Python entry or its reason.

The product promise is that each public Rust facade capability has a Python entry with the same
semantics. This check makes the promise mechanical without building anything. It reads

* `parity/public_api/antecedent.txt` (the committed rustdoc snapshot of the facade) and the
  `pub mod`s of `crates/antecedent/src/analysis/mod.rs` (the snapshot is regenerated deliberately
  and can lag the source),
* `parity/python_facade_surface.toml` (one `[[entry]]` per public free function and per public
  prepared or session type), and
* the Python sources under `python/` and the user documentation,

and fails when

  (a) a public facade free function or prepared / session type has no registry row, or a row names
      an entry that is no longer public (the registry must be regenerated deliberately);
  (b) a `bound` row's `python` path does not resolve to a definition under `python/`;
  (c) an `internal`, `missing` or `unverified` row carries no `reason`, or a `bound` row names no
      `python` path;
  (d) a documentation paragraph or registry detail says an entry has "no Python entry", is
      "Rust-only" or "not available from Python" while that same paragraph names a Rust entry whose
      row is `bound` (a stale sentence; reword it to cite the Python entry and its test).

`unverified` rows are reported, not failed: no Python counterpart was found by name and none was
confirmed by reading the wrappers. Run directly or via scripts/gate_public_api.sh.

    python3 scripts/check_python_surface.py
    python3 scripts/check_python_surface.py --self-test
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

from _repo import REPO_ROOT as ROOT

SNAPSHOT = ROOT / "parity/public_api/antecedent.txt"
REGISTRY = ROOT / "parity/python_facade_surface.toml"
STATUSES = {"bound", "internal", "missing", "unverified"}
KINDS = {"fn", "type"}
NEEDS_REASON = {"internal", "missing", "unverified"}

# A prepared or session type is an entry point of its own; builders and result types are not.
TYPE_SUFFIX = re.compile(r"::(Prepared[A-Za-z]*|[A-Za-z]*Session|[A-Za-z]*Study)$")
STALE = re.compile(
    r"(no|without a)\s+Python\s+(entry|wrapper|surface|binding)|Rust[- ]only|"
    r"not available from Python|None has a Python entry",
    re.IGNORECASE,
)
DOC_FILES = ("docs", "README.md")
DOC_EXTRA = ("parity/producer_consumer.toml", "parity/transport_stages.toml", "parity/promotion.toml")


def snapshot_entries(text: str) -> dict[str, str]:
    """Public free functions and prepared / session types of the facade snapshot."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") and not line.startswith("#["):
            continue
        if "#[doc(hidden)]" in line:
            continue
        line = re.sub(r"^(#\[[^\]]*\]\s*)+", "", line)
        match = re.match(r"pub (?:const )?fn (antecedent(?:::[a-z_0-9]+)+)[(<]", line)
        if match:
            entries[match.group(1)] = "fn"
            continue
        match = re.match(r"pub struct (antecedent(?:::[A-Za-z_0-9]+)+)", line)
        if match and TYPE_SUFFIX.search(match.group(1)):
            entries[match.group(1)] = "type"
    return entries


def public_items(text: str, item: str) -> list[str]:
    """Names of the top-level `pub` items matching `item` that are not `#[doc(hidden)]`.

    The attribute sits on its own line above the item, like the snapshot's inline marker, and
    a hidden item is not part of the public facade.
    """
    names: list[str] = []
    lines = text.splitlines()
    for number, line in enumerate(lines):
        match = re.match(rf"pub {item}", line)
        if match is None:
            continue
        above = number - 1
        hidden = False
        while above >= 0 and lines[above].lstrip().startswith(("#[", "///", "//")):
            hidden = hidden or "doc(hidden)" in lines[above]
            above -= 1
        if not hidden:
            names.append(match.group(1))
    return names


def source_entries(root: Path) -> dict[str, str]:
    """Public free functions and prepared / session types of the `pub mod`s of the analysis facade.

    The committed snapshot is regenerated deliberately and can lag the source; reading the source
    too means a new public module cannot ship without a row.
    """
    analysis = root / "crates/antecedent/src/analysis"
    modfile = analysis / "mod.rs"
    entries: dict[str, str] = {}
    if not modfile.is_file():
        return entries
    for module in re.findall(r"^pub mod ([a-z_0-9]+);", modfile.read_text(encoding="utf-8"), re.M):
        for file in (analysis / f"{module}.rs", analysis / module / "mod.rs"):
            if not file.is_file():
                continue
            text = file.read_text(encoding="utf-8")
            for name in public_items(text, r"(?:const )?fn ([a-z_0-9]+)[(<]"):
                entries[f"antecedent::analysis::{module}::{name}"] = "fn"
            for name in public_items(text, r"struct ([A-Za-z_0-9]+)"):
                if TYPE_SUFFIX.search(f"::{name}"):
                    entries[f"antecedent::analysis::{module}::{name}"] = "type"
    return entries


class PythonIndex:
    """Definitions under python/, found by parsing, never by importing."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[Path, ast.Module | None] = {}
        self._pyi: str | None = None
        self._rs: dict[Path, str] | None = None

    def _tree(self, path: Path) -> ast.Module | None:
        if path not in self._cache:
            try:
                self._cache[path] = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                self._cache[path] = None
        return self._cache[path]

    @classmethod
    def _names(cls, body: list[ast.stmt]) -> dict[str, ast.stmt]:
        names: dict[str, ast.stmt] = {}
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names[node.name] = node
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names[alias.asname or alias.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = node
            elif isinstance(node, ast.If | ast.Try):
                names.update(cls._names(node.body))
        return names

    def _module_file(self, dotted: list[str]) -> Path | None:
        base = self.root / "python"
        as_file = base.joinpath(*dotted).with_suffix(".py")
        if as_file.is_file():
            return as_file
        as_pkg = base.joinpath(*dotted) / "__init__.py"
        return as_pkg if as_pkg.is_file() else None

    def resolves(self, path: str) -> bool:
        if path.startswith("native:"):
            file = self.root / path.removeprefix("native:")
            return file.is_file()
        parts = path.split(".")
        if parts[:2] == ["antecedent", "_native"] and len(parts) == 3:
            if self._pyi is None:
                stub = self.root / "python/antecedent/_native.pyi"
                self._pyi = stub.read_text(encoding="utf-8") if stub.is_file() else ""
            return re.search(rf"^(def|class) {re.escape(parts[2])}\b", self._pyi, re.M) is not None
        # The longest module prefix that is a file, then names inside it.
        for split in range(len(parts) - 1, 0, -1):
            file = self._module_file(parts[:split])
            if file is None:
                continue
            tree = self._tree(file)
            if tree is None:
                return False
            body: list[ast.stmt] = tree.body
            for index, name in enumerate(parts[split:]):
                found = self._names(body).get(name)
                if found is None:
                    return False
                if index == len(parts[split:]) - 1:
                    return True
                if not isinstance(found, ast.ClassDef):
                    return False
                body = found.body
            return True
        return False


def load_registry(path: Path) -> list[dict[str, str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return list(data.get("entry", []))


def registry_problems(
    entries: dict[str, str], rows: list[dict[str, str]], index: PythonIndex
) -> tuple[list[str], Counter[str]]:
    problems: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        rust, kind, status = row.get("rust", ""), row.get("kind", ""), row.get("status", "")
        if rust in seen:
            problems.append(f"{rust}: duplicate row")
        seen.add(rust)
        counts[status] += 1
        if kind not in KINDS or status not in STATUSES:
            problems.append(f"{rust}: kind {kind!r} / status {status!r} is not recognised")
            continue
        if rust not in entries:
            problems.append(f"{rust}: row names an entry that is not in the facade snapshot")
        elif entries[rust] != kind:
            problems.append(f"{rust}: row kind {kind} but the snapshot has a {entries[rust]}")
        if status == "bound":
            python = row.get("python", "")
            if not python:
                problems.append(f"{rust}: bound row names no python entry")
            elif not index.resolves(python):
                problems.append(f"{rust}: python entry {python} does not resolve under python/")
        elif status in NEEDS_REASON and not row.get("reason", "").strip():
            problems.append(f"{rust}: {status} row states no reason")
    for rust in sorted(set(entries) - seen):
        problems.append(f"{rust}: public facade entry has no row in {REGISTRY.name}")
    return problems, counts


def doc_units(root: Path) -> list[tuple[str, int, str]]:
    """(file, first line, text) for each paragraph of the docs and each registry row."""
    files: list[Path] = []
    for name in DOC_FILES:
        target = root / name
        if target.is_dir():
            files += [p for p in sorted(target.rglob("*.md")) if "release-notes" not in p.parts]
        elif target.is_file():
            files.append(target)
    files += [root / name for name in DOC_EXTRA if (root / name).is_file()]
    units: list[tuple[str, int, str]] = []
    for file in files:
        rel = file.relative_to(root).as_posix()
        block: list[str] = []
        start = 1
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            table_row = line.lstrip().startswith("|") or file.suffix == ".toml"
            if not line.strip() or table_row:
                if block:
                    units.append((rel, start, " ".join(block)))
                    block = []
                if table_row and line.strip():
                    units.append((rel, number, line))
                continue
            if not block:
                start = number
            block.append(line.strip())
        if block:
            units.append((rel, start, " ".join(block)))
    return units


def stale_doc_problems(rows: list[dict[str, str]], units: list[tuple[str, int, str]]) -> list[str]:
    bound = {}
    for row in rows:
        if row.get("status") == "bound" and row.get("kind") == "fn":
            segments = row["rust"].split("::")
            name = segments[-1]
            # A short or generic name is only recognised with its module.
            if name.count("_") >= 1 and len(name) >= 12:
                bound[name] = row
    problems: list[str] = []
    for rel, line, text in units:
        if not STALE.search(text):
            continue
        for name, row in bound.items():
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
                phrase = STALE.search(text)
                assert phrase is not None
                problems.append(
                    f"{rel}:{line}: says {phrase.group(0)!r} but names `{name}`, which "
                    f"{row['python']} carries; cite the Python entry and its test"
                )
    return problems


def run(root: Path = ROOT) -> int:
    entries = snapshot_entries((root / SNAPSHOT.relative_to(ROOT)).read_text(encoding="utf-8"))
    # An item the snapshot already lists under another public path is one entry, not two.
    known = {path.rsplit("::", 1)[-1] for path in entries}
    entries |= {k: v for k, v in source_entries(root).items() if k.rsplit("::", 1)[-1] not in known}
    rows = load_registry(root / REGISTRY.relative_to(ROOT))
    problems, counts = registry_problems(entries, rows, PythonIndex(root))
    problems += stale_doc_problems(rows, doc_units(root))
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    if problems:
        print("Python facade surface check FAILED:\n" + "\n".join(f" - {p}" for p in problems))
        return 1
    print(f"Python facade surface OK ({len(entries)} entries: {summary})")
    unverified = sorted(r["rust"] for r in rows if r.get("status") == "unverified")
    if unverified:
        print(f"  {len(unverified)} unverified (not failed): " + ", ".join(unverified[:5]) + " ...")
    return 0


def self_test() -> int:
    snapshot = (
        "pub fn antecedent::analysis::tier_diagnostics::read_tier_diagnostics(x: i32)\n"
        "pub struct antecedent::PreparedThing\n"
        "pub struct antecedent::Plain\n"
    )
    entries = snapshot_entries(snapshot)
    assert entries == {
        "antecedent::analysis::tier_diagnostics::read_tier_diagnostics": "fn",
        "antecedent::PreparedThing": "type",
    }, entries
    index = PythonIndex(ROOT)
    rows = [
        {"rust": "antecedent::PreparedThing", "kind": "type", "status": "internal"},
        {
            "rust": "antecedent::analysis::tier_diagnostics::read_tier_diagnostics",
            "kind": "fn",
            "status": "bound",
            "python": "antecedent.tier_diagnostics.tier_diagnostics",
        },
    ]
    problems, _ = registry_problems(entries, rows, index)
    assert any("internal row states no reason" in p for p in problems), problems
    stale = stale_doc_problems(
        rows,
        [("docs/x.md", 3, "The route `read_tier_diagnostics` is Rust-only; no Python wrapper.")],
    )
    assert len(stale) == 1 and "docs/x.md:3" in stale[0], stale
    assert not stale_doc_problems(rows, [("docs/x.md", 1, "A Rust-only helper `other_helper_name`.")])
    hidden_source = "#[doc(hidden)]\npub fn sealed_helper(x: i32) {}\n\n/// Doc.\npub fn open_helper() {}\n"
    assert public_items(hidden_source, r"(?:const )?fn ([a-z_0-9]+)[(<]") == ["open_helper"]
    assert snapshot_entries("#[doc(hidden)] pub fn antecedent::gcm::hidden_one(x: i32)\n") == {}
    print("check_python_surface self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv[1:] else run())
