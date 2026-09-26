#!/usr/bin/env python3
"""Which coverage records a change to the statistical surface invalidates.

A coverage record in `parity/coverage_records.toml` stands until the code it
measured changes. `scripts/calibration_surface.list` is the only owner of that
surface. It assigns every path to a facet:

* `core` — shared numerical and harness code (least-squares, conjugate,
  bootstrap, RNG, facade dispatch, manifests, toolchain). Every record
  carries it: a core edit owes a re-measurement unless a reviewed replay
  waiver covers the change.
* `estimator.*` / `identity.*` — the implementation of one estimator or
  identification path. A change owes only the records whose `estimator` /
  `query` (or other keyed field) names that implementation.
* `suite.*` / `mechanism` / `design` — as before: only the records that
  carry the facet.

A record's facets are derived from the record itself, never declared by hand:
the facet of the file its `test` and `dgp` name, and every facet a `key`
line in the list assigns to one of its fields.

A narrow facet is only sound while code outside it cannot run its code on
behalf of records that do not carry it. `check` enforces that boundary from the
source: a file outside the facet that names one of the facet's items — through
a facet crate's path (`antecedent_model::…`, which the compiler makes the only
way in) or through an item a facet file defines — fails unless an `allow` line
lists exactly those names for that file. An `allow` line is a reviewed claim
that the reference cannot move a measured number (an error conversion, a result
field type, a query-keyed dispatch arm whose query a `key` line names); a new
reference or a stale entry fails until someone reviews it. Name matching is
deliberately over-inclusive: an item name the facet shares with other code
counts as a reference, which can only ask for more review, never less.

A drifted record can instead be `attested_by_replay` under a replay waiver in
`parity/calibration_waivers.toml` (outside the surface): a reviewed claim that
named paths changed without moving a number, backed by records re-run at the
waiver's `to` that reproduced the stored records bit for bit. Its scope is
mechanical (see `waiver_applies`); it is reported under its own status, and a
waiver that fails validation attests nothing and fails `check`.

Usage:

    python3 scripts/calibration_facets.py check            # list + boundaries + waivers
    python3 scripts/calibration_facets.py status           # drift per measured SHA
    python3 scripts/calibration_facets.py status --require # the gate: every record must stand
    python3 scripts/calibration_facets.py counts           # records carrying each facet
    python3 scripts/calibration_facets.py stale-tests      # tests whose records owe a re-measurement
    python3 scripts/calibration_facets.py replay-candidates --from <ref> [--to <ref>] [path...]
    python3 scripts/calibration_facets.py replay --waiver <id> [--dry-run]   # at the waiver's `to`
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import struct
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "scripts" / "calibration_surface.list"
RECORDS = ROOT / "parity" / "coverage_records.toml"
GATES = ROOT / "parity" / "calibration_gates.toml"
# Ids of the pseudo-records the record-less group ledger contributes: a group that
# passed at a commit is attested exactly like a record measured there, but never by
# a replay waiver (a replay reproduces stored records; a ledger row stores none).
GATE_PREFIX = "gate."

CORE = "core"
FACET_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)?")
RECORD_KEY_FIELDS = (
    "query",
    "graph_class",
    "structure",
    "modality",
    "inference",
    "estimator",
    "interval_method",
    "se_kind",
    "dependence",
    "posterior",
    "functional",
    "identification",
)
# Manifests whose workspace-version lines a release bump rewrites. Their other
# content (features, external dependency versions, profiles) is compared.
NORMALIZED = re.compile(r"(^|/)Cargo\.(toml|lock)$")
MAX_PATHS_SHOWN = 12


# --------------------------------------------------------------------------
# Rust source: comments and literals removed, so names are code, not prose.
# --------------------------------------------------------------------------

_SPECIAL = re.compile(r"//|/\*|b?r#*\"|\"|'")
_CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:(?:unsafe|async|const|extern|default)[ \t]+)*"
    r"(?:fn|struct|enum|trait|type|const|static|union)[ \t]+([A-Za-z_][A-Za-z0-9_]*)"
    r"|macro_rules![ \t]*([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)


def rust_code(text: str) -> str:
    """`text` with comments dropped and string / char literal contents blanked."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _SPECIAL.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        start = m.start()
        tok = m.group(0)
        if tok.startswith(("r", "b")) and start > 0 and (
            text[start - 1].isalnum() or text[start - 1] == "_"
        ):
            # `for"` / `abr"`: an identifier that ends in r or b, then a quote.
            out.append(text[i : start + len(tok) - 1])
            i = start + len(tok) - 1
            continue
        out.append(text[i:start])
        if tok == "//":
            end = text.find("\n", start)
            i = n if end < 0 else end
        elif tok == "/*":
            depth, j = 1, start + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    if text[j] == "\n":
                        out.append("\n")
                    j += 1
            i = j
        elif tok == '"':
            j = start + 1
            while j < n and text[j] != '"':
                if text[j] == "\n":
                    out.append("\n")
                j += 2 if text[j] == "\\" else 1
            out.append('""')
            i = j + 1
        elif tok == "'":
            ch = _CHAR.match(text, start)
            if ch:
                out.append("' '")
                i = ch.end()
            else:  # a lifetime or label
                out.append("'")
                i = start + 1
        else:  # raw string r#"…"#
            hashes = tok.count("#")
            end = text.find('"' + "#" * hashes, m.end())
            end = n if end < 0 else end + 1 + hashes
            out.append('""' + "\n" * text.count("\n", start, end))
            i = end
    return "".join(out)



def rust_mask(text: str) -> str:
    """`text` with comments and string / char literals blanked to spaces, keeping every
    offset and newline, so braces found in the mask are braces of the code."""
    out: list[str] = []
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        out.append("".join("\n" if c == "\n" else " " for c in text[a:b]))

    while i < n:
        m = _SPECIAL.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        start = m.start()
        tok = m.group(0)
        if tok.startswith(("r", "b")) and start > 0 and (
            text[start - 1].isalnum() or text[start - 1] == "_"
        ):
            out.append(text[i : start + len(tok) - 1])
            i = start + len(tok) - 1
            continue
        out.append(text[i:start])
        if tok == "//":
            end = text.find("\n", start)
            end = n if end < 0 else end
            blank(start, end)
            i = end
        elif tok == "/*":
            depth, j = 1, start + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            j = min(j, n)
            blank(start, j)
            i = j
        elif tok == '"':
            j = start + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, n)
            blank(start, end)
            i = end
        elif tok == "'":
            ch = _CHAR.match(text, start)
            if ch:
                blank(start, ch.end())
                i = ch.end()
            else:  # a lifetime or label
                out.append("'")
                i = start + 1
        else:  # raw string r#"..."#
            hashes = tok.count("#")
            end = text.find('"' + "#" * hashes, m.end())
            end = n if end < 0 else end + 1 + hashes
            blank(start, end)
            i = end
    return "".join(out)


_TEST_MOD = re.compile(
    r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{"
)


def strip_test_modules(text: str) -> str:
    """`text` without its inline `#[cfg(test)] mod name { ... }` blocks.

    Only inline test modules go; a `#[cfg(test)]` item of any other kind, and an
    out-of-line `mod tests;` (a different file), stay."""
    mask = rust_mask(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    while (m := _TEST_MOD.search(mask, pos)) is not None:
        depth, j = 1, m.end()
        while j < len(mask) and depth:
            depth += (mask[j] == "{") - (mask[j] == "}")
            j += 1
        spans.append((m.start(), j))
        pos = j
    out: list[str] = []
    last = 0
    for a, b in spans:
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return "".join(out)


def declares_out_of_line_test_module(rel: str, root: Path | None = None) -> bool:
    """Whether `rel` is a file that a sibling module declares as `#[cfg(test)] mod stem;`,
    so the whole file is compiled only into unit-test builds."""
    root = ROOT if root is None else root
    path = Path(rel)
    stem = path.stem
    parent = root / path.parent
    candidates = [parent / "lib.rs", parent / "mod.rs", parent / "main.rs"]
    candidates.append(root / path.parent.parent / f"{path.parent.name}.rs")
    pattern = re.compile(
        r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+"
        + re.escape(stem)
        + r"\s*;"
    )
    for cand in candidates:
        if cand.is_file() and pattern.search(rust_mask(cand.read_text(errors="ignore"))):
            return True
    return False


def scaffolding_view(rel: str, text: str) -> str:
    """`text` as the calibration surface sees a testmod file: without its inline test
    modules, or empty when the whole file is an out-of-line `#[cfg(test)]` module."""
    return "" if declares_out_of_line_test_module(rel) else strip_test_modules(text)

_PUB_USE = re.compile(r"^[ \t]*pub(?:\([^)]*\))?[ \t]+use\b([^;]*);", re.M)


_IMPL_OR_TRAIT = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:unsafe[ \t]+)?(?:impl|trait)\b", re.M
)


def strip_impl_bodies(code: str) -> str:
    """`code` with the bodies of `impl` and `trait` blocks blanked.

    A method, associated function, associated type or associated const lives
    inside one of these blocks and is reached only through the type or trait
    that owns it, never by its bare name, so a `new`, `fmt` or `query` defined
    there is not an item another file could name."""
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        m = _IMPL_OR_TRAIT.search(code, i)
        if m is None:
            out.append(code[i:])
            break
        # The block opens at the first `{` outside the header's generics and
        # bounds (`impl<T: Fn() -> X> Y for Z {`); a `trait T;`-like header
        # without a body (`impl Trait for T;` is not Rust, but stay safe) ends at `;`.
        j, depth = m.end(), 0
        while j < n and not (code[j] == "{" and depth == 0) and code[j] != ";":
            if code.startswith("->", j):
                j += 2
                continue
            depth += {"<": 1, ">": -1}.get(code[j], 0)
            j += 1
        if j >= n or code[j] == ";":
            out.append(code[i : j + 1])
            i = j + 1
            continue
        k, depth = j + 1, 1
        while k < n and depth:
            depth += {"{": 1, "}": -1}.get(code[k], 0)
            k += 1
        out.append(code[i : j + 1] + re.sub(r"[^\n]", " ", code[j + 1 : k - 1]) + code[k - 1 : k])
        i = k
    return "".join(out)


def definitions(code: str) -> set[str]:
    """Items a file defines, plus every name it re-exports with `pub use`.

    Methods and other associated items are not items (see `strip_impl_bodies`), and
    neither is the scaffolding of an inline `#[cfg(test)] mod`, which no other file
    can name."""
    names = {a or b for a, b in _DEF.findall(strip_impl_bodies(strip_test_modules(code)))}
    for m in _PUB_USE.finditer(code):
        names |= set(_IDENT.findall(m.group(1))) - {"self", "super", "crate", "as"}
    return names


def anchored_names(code: str, anchor: str) -> tuple[set[str], bool]:
    """Names in path expressions that start at `anchor`, and whether any re-exports."""
    names: set[str] = set()
    reexport = False
    for m in re.finditer(rf"\b{re.escape(anchor)}\b", code):
        line_start = code.rfind("\n", 0, m.start()) + 1
        if re.match(r"[ \t]*pub(?:\([^)]*\))?[ \t]+use\b", code[line_start : m.start()]):
            reexport = True
        j, n = m.end(), len(code)
        while j < n:
            while j < n and code[j] in " \t\n":
                j += 1
            if not code.startswith("::", j):
                break
            j += 2
            while j < n and code[j] in " \t\n":
                j += 1
            if j < n and code[j] == "{":
                depth, k = 1, j + 1
                while k < n and depth:
                    depth += {"{": 1, "}": -1}.get(code[k], 0)
                    k += 1
                names |= set(_IDENT.findall(code[j:k])) - {"self", "super", "crate", "as"}
                if "*" in code[j:k]:
                    names.add("*")
                break
            if j < n and code[j] == "*":
                names.add("*")
                break
            ident = _IDENT.match(code, j)
            if not ident:
                break
            names.add(ident.group(0))
            j = ident.end()
    return names, reexport


# --------------------------------------------------------------------------
# The list.
# --------------------------------------------------------------------------


@dataclass
class Surface:
    entries: list[tuple[str, str]] = field(default_factory=list)  # (facet, path)
    allows: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    testmods: set[str] = field(default_factory=set)
    keys: list[tuple[str, str, list[str]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [path for _, path in self.entries]

    @property
    def facets(self) -> set[str]:
        return {facet for facet, _ in self.entries}

    def facet_of(self, rel: str) -> str | None:
        """Facet of the longest line covering `rel`; None outside the surface."""
        best: tuple[int, str] | None = None
        for facet, path in self.entries:
            covered = rel == path or (path.endswith("/") and rel.startswith(path))
            if covered and (best is None or len(path) > best[0]):
                best = (len(path), facet)
        return None if best is None else best[1]

    def members(self, facet: str, files: list[str]) -> list[str]:
        return [f for f in files if self.facet_of(f) == facet]


def load_surface(path: Path = LIST, text: str | None = None) -> Surface:
    surface = Surface()
    seen: set[str] = set()
    text = path.read_text() if text is None else text
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        where = f"{path.name}:{lineno}"
        if parts[0] == "allow":
            if len(parts) < 4:
                surface.errors.append(f"{where}: allow <facet> <path> <name>...")
                continue
            key = (parts[1], parts[2])
            if key in surface.allows:
                surface.errors.append(f"{where}: duplicate allow for {parts[1]} {parts[2]}")
            surface.allows[key] = set(parts[3:])
        elif parts[0] == "testmod":
            if len(parts) != 2 or not parts[1].endswith(".rs"):
                surface.errors.append(f"{where}: testmod <path to a .rs file>")
                continue
            if parts[1] in surface.testmods:
                surface.errors.append(f"{where}: duplicate testmod for {parts[1]}")
            surface.testmods.add(parts[1])
        elif parts[0] == "key":
            if len(parts) < 4 or parts[2] not in RECORD_KEY_FIELDS:
                surface.errors.append(
                    f"{where}: key <facet> <{'|'.join(RECORD_KEY_FIELDS)}> <pattern>..."
                )
                continue
            surface.keys.append((parts[1], parts[2], parts[3:]))
        else:
            if len(parts) != 2:
                surface.errors.append(f"{where}: expected '<facet> <path>', got {line!r}")
                continue
            facet, rel = parts
            if not FACET_NAME.fullmatch(facet):
                surface.errors.append(f"{where}: bad facet name {facet!r}")
            if rel in seen:
                surface.errors.append(f"{where}: {rel} listed twice")
            seen.add(rel)
            target = ROOT / rel
            if rel.endswith("/") and not target.is_dir():
                surface.errors.append(f"{where}: {rel} is not a directory")
            elif not rel.endswith("/") and not target.is_file():
                surface.errors.append(f"{where}: {rel} is not a file")
            surface.entries.append((facet, rel))
    known = surface.facets
    if CORE not in known:
        surface.errors.append(f"{path.name}: no {CORE} facet")
    for facet, rel in surface.allows:
        if facet not in known or facet == CORE:
            surface.errors.append(f"{path.name}: allow names unknown or core facet {facet}")
    for facet, _, _ in surface.keys:
        if facet not in known or facet == CORE:
            surface.errors.append(f"{path.name}: key names unknown or core facet {facet}")
    return surface


# --------------------------------------------------------------------------
# Boundaries.
# --------------------------------------------------------------------------


def _crate_deps() -> dict[str, set[str]]:
    deps: dict[str, set[str]] = {}
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        data = tomllib.loads(manifest.read_text())
        names: set[str] = set()
        tables = [data.get(t, {}) for t in ("dependencies", "dev-dependencies", "build-dependencies")]
        for target in data.get("target", {}).values():
            tables += [target.get(t, {}) for t in ("dependencies", "dev-dependencies")]
        for table in tables:
            names |= {name for name in table if name.startswith("antecedent")}
        deps[manifest.parent.name] = names
    # Transitive: a crate reaches everything its dependencies reach.
    changed = True
    while changed:
        changed = False
        for crate, direct in deps.items():
            closure = set(direct)
            for dep in direct:
                closure |= deps.get(dep, set())
            if closure != direct:
                deps[crate] = closure
                changed = True
    return deps


def _crate(rel: str) -> str | None:
    parts = rel.split("/")
    return parts[1] if parts[0] == "crates" and len(parts) > 2 else None


def _test_only_module(rel: str) -> bool:
    """A library file its parent declares as `#[cfg(test)] mod <stem>;`."""
    path = ROOT / rel
    if "/src/" not in rel or path.name in ("lib.rs", "mod.rs"):
        return False
    for parent in (path.parent / "lib.rs", path.parent / "mod.rs", path.parent.with_suffix(".rs")):
        if parent.is_file() and re.search(
            rf"#\[cfg\(test\)\](?:\s*#\[[^\]]+\])*\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+{re.escape(path.stem)}\s*;",
            _code(parent.relative_to(ROOT).as_posix()),
        ):
            return True
    return False


def _unit(rel: str) -> str:
    """Compilation unit: a crate library, one integration-test crate, a test harness
    compiled into every test crate, or a `#[cfg(test)]` module nothing else names."""
    parts = rel.split("/")
    if parts[0] != "crates" or len(parts) < 3:
        return rel
    if parts[2] == "tests":
        if len(parts) > 4 and parts[3] == "common":
            return f"crates/{parts[1]}/tests/common"
        return f"crates/{parts[1]}/tests/{Path(parts[-1]).stem}"
    return rel if _test_only_module(rel) else parts[1]


def is_test_consumer(rel: str) -> bool:
    """Suite and harness files: their references tag records instead of failing."""
    return "/tests/" in rel or _test_only_module(rel)


def _reaches(user: str, member: str, deps: dict[str, set[str]]) -> bool:
    """Can code in `user` name an item `member` defines?"""
    unit_user, unit_member = _unit(user), _unit(member)
    if unit_user == unit_member:
        return True
    crate_user, crate_member = _crate(user), _crate(member)
    if unit_member == crate_member:  # a crate library
        return crate_user is not None and (
            crate_member == crate_user or crate_member in deps.get(crate_user, set())
        )
    if unit_member.endswith("/tests/common"):  # compiled into every test crate of the crate
        return unit_user.startswith(unit_member.rsplit("/", 1)[0] + "/")
    return False


def surface_rust_files(surface: Surface) -> list[str]:
    files: set[str] = set()
    for rel in surface.paths:
        target = ROOT / rel
        if rel.endswith("/"):
            files |= {p.relative_to(ROOT).as_posix() for p in target.rglob("*.rs")}
        elif rel.endswith(".rs"):
            files.add(rel)
    return sorted(files)


@dataclass
class References:
    """`by_facet[facet][file]` = names `file` uses from `facet`; plus re-exports."""

    by_facet: dict[str, dict[str, set[str]]]
    reexports: list[tuple[str, str]]
    harness: list[str]


_CODE_CACHE: dict[str, str] = {}


def _code(rel: str) -> str:
    if rel not in _CODE_CACHE:
        _CODE_CACHE[rel] = rust_code((ROOT / rel).read_text(errors="ignore"))
    return _CODE_CACHE[rel]


def references(surface: Surface) -> References:
    files = surface_rust_files(surface)
    deps = _crate_deps()
    facet_of = {rel: surface.facet_of(rel) for rel in files}
    defs = {rel: definitions(_code(rel)) for rel in files}
    tokens = {rel: set(_IDENT.findall(_code(rel))) for rel in files}
    by_facet: dict[str, dict[str, set[str]]] = {}
    reexports: list[tuple[str, str]] = []
    for facet in sorted(surface.facets - {CORE}):
        if _keyed_impl_facet(facet):
            # Isolation is the key line, not an allow-list: crate lib.rs re-exports
            # every estimator/identity, which would otherwise fail check.
            by_facet[facet] = {}
            continue
        members = [rel for rel in files if facet_of[rel] == facet]
        # Whole crates in the facet: reached only through their crate path.
        whole = sorted(
            match.group(1)
            for facet_, rel in surface.entries
            if facet_ == facet and (match := re.fullmatch(r"crates/([^/]+)/src/", rel))
        )
        anchors = {crate.replace("-", "_"): crate for crate in whole}
        # Facet files inside a crate that is not in the facet: reached by name.
        partial = [rel for rel in members if _crate(rel) not in whole]
        hits: dict[str, set[str]] = {}
        for rel in files:
            if facet_of[rel] == facet:
                continue
            crate = _crate(rel)
            reach = deps.get(crate or "", set()) | ({crate} if crate else set())
            found: set[str] = set()
            for anchor, anchor_crate in anchors.items():
                if anchor_crate in reach and anchor in tokens[rel]:
                    used, reexport = anchored_names(_code(rel), anchor)
                    found |= used | {anchor}
                    if reexport:
                        reexports.append((facet, rel))
            for member in partial:
                if not _reaches(rel, member, deps):
                    continue
                # Every item the file defines, matched by bare name (a shared
                # name counts), and the module itself wherever a path or a
                # `mod` declaration names it.
                found |= tokens[rel] & defs[member]
                stem = Path(member).stem
                used, _ = anchored_names(_code(rel), stem)
                if used or re.search(rf"\bmod\s+{re.escape(stem)}\b", _code(rel)):
                    found |= used | {stem}
            if found:
                hits[rel] = found
        by_facet[facet] = hits
    harness = [rel for rel in files if "/tests/common/" in rel]
    return References(by_facet, reexports, harness)


def _keyed_impl_facet(facet: str) -> bool:
    return facet.startswith(("estimator.", "identity."))


def check(surface: Surface) -> list[str]:
    problems = list(surface.errors)
    if problems:
        return problems
    # Completeness: nothing that can move a number sits outside the surface.
    required = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "scripts/gate_calibration.sh",
        "crates/antecedent/tests/common/calibration.rs",
    ]
    for crate in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        base = crate.parent.relative_to(ROOT).as_posix()
        required += [f"{base}/Cargo.toml", f"{base}/src/lib.rs"]
        if (crate.parent / "build.rs").is_file():
            required.append(f"{base}/build.rs")
    tests = ROOT / "crates/antecedent/tests"
    for suite in sorted(tests.glob("*.rs")):
        text = suite.read_text(errors="ignore")
        if suite.name.startswith(("v19_", "v110_calibration_")) or "for_record(" in text:
            required.append(suite.relative_to(ROOT).as_posix())
    for harness in sorted((tests / "common").rglob("*")):
        if harness.is_file():
            required.append(harness.relative_to(ROOT).as_posix())
    for rel in required:
        if surface.facet_of(rel) is None:
            problems.append(f"calibration surface omits {rel}")
    for rel in ("scripts/gate_calibration.sh", "crates/antecedent/tests/common/calibration.rs"):
        if surface.facet_of(rel) not in (None, CORE):
            problems.append(f"{rel} must be {CORE}: the shared harness is not an estimator")
    # Boundaries.
    refs = references(surface)
    for facet, rel in refs.reexports:
        if _keyed_impl_facet(facet):
            continue
        problems.append(
            f"{rel} re-exports {facet} items from outside the facet; make it a {facet} "
            "file or stop re-exporting"
        )
    for facet, hits in sorted(refs.by_facet.items()):
        if _keyed_impl_facet(facet):
            continue
        for rel, found in sorted(hits.items()):
            if is_test_consumer(rel):
                continue
            allowed = surface.allows.get((facet, rel), set())
            if found - allowed:
                problems.append(
                    f"{rel} names {facet} items outside the facet: "
                    f"{' '.join(sorted(found - allowed))} (review, then add or extend "
                    f"`allow {facet} {rel} ...` in scripts/calibration_surface.list)"
                )
            if allowed - found:
                problems.append(
                    f"allow {facet} {rel}: stale names {' '.join(sorted(allowed - found))}"
                )
    for (facet, rel), _ in sorted(surface.allows.items()):
        if rel not in refs.by_facet.get(facet, {}):
            problems.append(f"allow {facet} {rel}: the file no longer names {facet} items")
    hosted = {
        str(spec).rsplit("::", 1)[0]
        for rec in load_records()
        for spec in (rec.get("test", ""), rec.get("dgp", ""))
        if spec
    }
    for rel in sorted(surface.testmods):
        facet = surface.facet_of(rel)
        if facet is None:
            problems.append(f"testmod {rel}: not a path of the calibration surface")
        elif facet.startswith("suite.") or rel in hosted:
            problems.append(
                f"testmod {rel}: a record's test or dgp lives in this file, so its test "
                "modules are measurement, not scaffolding"
            )
    return problems


# --------------------------------------------------------------------------
# Records.
# --------------------------------------------------------------------------


def load_records(path: Path = RECORDS) -> list[dict]:
    if not path.is_file():
        return []
    return tomllib.loads(path.read_text()).get("record", [])


def load_gate_rows(path: Path = GATES) -> list[dict]:
    """The record-less calibration groups (CI Type I, uniformity, SBC, ...) as pseudo-records.

    Their logs leave no coverage record, so without this a change to the core facet could
    never make one of them owe a re-run. Each depends on `core` only: the ledger does not
    name the suite behind a group, and over-attesting nothing is the safe direction."""
    if not path.is_file():
        return []
    return [
        {
            "id": f"{GATE_PREFIX}{row['group']}",
            "test": "",
            "dgp": "",
            "calibration_sha": row["calibration_sha"],
        }
        for row in tomllib.loads(path.read_text()).get("gate", [])
    ]


def record_facets(rec: dict, surface: Surface, refs: References | None = None) -> list[str]:
    """Facets a record depends on, derived from its own fields.

    A record carries `core`, the suite of its test/DGP, and every keyed
    estimator or identity its fields name.
    """
    del refs
    facets: set[str] = {CORE}
    for spec in (str(rec.get("test", "")), str(rec.get("dgp", ""))):
        rel = spec.rsplit("::", 1)[0]
        facet = surface.facet_of(rel)
        if facet:
            facets.add(facet)
    for facet, key, patterns in surface.keys:
        value = str(rec.get(key, ""))
        if any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns):
            facets.add(facet)
    return sorted(facets)


# --------------------------------------------------------------------------
# Drift.
# --------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)


def resolve(sha: str) -> str | None:
    out = _git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    return out.stdout.strip() if out.returncode == 0 else None


def _normalized_manifest(text: str, lock: bool) -> str:
    data = tomllib.loads(text)
    if lock:
        data["package"] = [
            {k: v for k, v in pkg.items() if k != "dependencies"}
            | {"dependencies": [d.split(" ")[0] for d in pkg.get("dependencies", [])]}
            for pkg in data.get("package", [])
            if not str(pkg.get("name", "")).startswith("antecedent")
        ]
        data.pop("version", None)
        return json.dumps(data, sort_keys=True)

    def strip_versions(table: dict) -> None:
        for name, spec in table.items():
            if name.startswith("antecedent") and isinstance(spec, dict):
                spec.pop("version", None)

    for section in ("package", "workspace"):
        node = data.get(section, {})
        node.pop("version", None)
        if isinstance(node.get("package"), dict):
            node["package"].pop("version", None)
        if isinstance(node.get("dependencies"), dict):
            strip_versions(node["dependencies"])
    for t in ("dependencies", "dev-dependencies", "build-dependencies"):
        strip_versions(data.get(t, {}))
    for target in data.get("target", {}).values():
        for t in ("dependencies", "dev-dependencies", "build-dependencies"):
            strip_versions(target.get(t, {}))
    return json.dumps(data, sort_keys=True)


def changed_paths(surface: Surface, sha: str, head: str | None = None) -> list[str]:
    """Surface paths whose content differs between `sha` and `head` (default: the
    working tree), with workspace version numbers ignored in the manifests."""
    between = [sha] if head is None else [sha, head]
    # NUL-separated: a path with a space (or one git would quote) stays one path.
    diff = _git("diff", "--name-only", "-z", *between, "--", *surface.paths)
    if diff.returncode != 0:
        raise SystemExit(f"git diff {' '.join(between)} failed: {diff.stderr.strip()}")
    changed = {p for p in diff.stdout.split("\0") if p}
    if head is None:
        untracked = _git("ls-files", "--others", "--exclude-standard", "-z", "--", *surface.paths)
        changed |= {p for p in untracked.stdout.split("\0") if p}
    kept = []
    for rel in sorted(changed):
        if rel in surface.testmods:
            old = _git("show", f"{sha}:{rel}")
            if head is None:
                new_text = (ROOT / rel).read_text() if (ROOT / rel).is_file() else None
            else:
                new = _git("show", f"{head}:{rel}")
                new_text = new.stdout if new.returncode == 0 else None
            if (
                old.returncode == 0
                and new_text is not None
                and scaffolding_view(rel, old.stdout) == scaffolding_view(rel, new_text)
            ):
                continue
        if NORMALIZED.search(rel):
            old = _git("show", f"{sha}:{rel}")
            if head is None:
                new_text = (ROOT / rel).read_text() if (ROOT / rel).is_file() else None
            else:
                new = _git("show", f"{head}:{rel}")
                new_text = new.stdout if new.returncode == 0 else None
            if old.returncode == 0 and new_text is not None:
                lock = rel.endswith(".lock")
                try:
                    same = _normalized_manifest(old.stdout, lock) == _normalized_manifest(
                        new_text, lock
                    )
                except tomllib.TOMLDecodeError:
                    same = False
                if same:
                    continue
        kept.append(rel)
    return kept


class Repo:
    """Commits and surface drift, read from git."""

    def __init__(self) -> None:
        self._resolved: dict[str, str | None] = {}
        self._changed: dict[tuple[str, str | None], list[str]] = {}

    def resolve(self, ref: str) -> str | None:
        if ref not in self._resolved:
            self._resolved[ref] = resolve(ref) if ref else None
        return self._resolved[ref]

    def changed(self, surface: Surface, sha: str, head: str | None = None) -> list[str]:
        """Surface paths differing between commit `sha` and `head` (None: the worktree)."""
        if (sha, head) not in self._changed:
            self._changed[(sha, head)] = changed_paths(surface, sha, head)
        return list(self._changed[(sha, head)])


class FakeRepo(Repo):
    """A repository described by a table, for the self-test."""

    def __init__(self, commits: dict[str, str], diffs: dict[tuple[str, str | None], list[str]]):
        super().__init__()
        self.commits = commits
        self.diffs = diffs

    def resolve(self, ref: str) -> str | None:
        return self.commits.get(ref)

    def changed(self, surface: Surface, sha: str, head: str | None = None) -> list[str]:
        return list(self.diffs.get((sha, head), []))


# --------------------------------------------------------------------------
# Replay waivers: a reviewed, non-numeric change accepted with evidence.
# --------------------------------------------------------------------------
#
# A waiver in parity/calibration_waivers.toml (outside the surface) names the
# commit a set of records was measured at (`from`), the commit it attests
# forward to (`to`), and the exact surface `paths` that changed between them.
# It carries replay evidence: records measured at `from`, re-run at `to`
# through the unchanged gate, whose emitted `calibration-record` payloads were
# bit-identical to the stored ones. A record owing a re-measurement is
# `attested_by_replay` under a valid waiver only when it was measured at
# `from`, every path in its facets that changed since is one the waiver names,
# and none of those facets changed between `to` and the tree. Anything else
# still owes, and a waiver that fails validation attests nothing and fails
# `check`.

WAIVERS = ROOT / "parity" / "calibration_waivers.toml"
WAIVER_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
# Stored record fields that are bookkeeping, not measurement output.
NOT_EMITTED = frozenset({"calibration_sha", "facets", "surface_list_blob"})
LOG_DIR = ROOT / "target" / "calibration-records"
REPLAY_DIR = ROOT / "target" / "calibration-replay"
WAIVERS_HEADER = """\
# Replay waivers for coverage records (scripts/calibration_facets.py).
#
# A waiver lets a reviewed change that cannot move a measured number stand in
# for a re-measurement, with evidence. It applies to a record only when the
# record was measured at `from`, every attested path in the record's facets
# that changed since is listed in `paths`, and none of them changed after
# `to`. Its evidence is a replay: the `replay` records, measured at `from`,
# re-run at `to` through scripts/gate_calibration.sh and compared bit for bit
# with the stored records. Each `exercises` list names the waived paths that
# record's test runs; together they must cover every path.
#
# Write `id`, `from`, `to`, `reviewed_by`, `justification`, `paths` and the
# `replay` tables by hand, then fill `outcome` by running
#   python3 scripts/calibration_facets.py replay --waiver <id>
# on a clean checkout of `to`. That command rewrites this file; comments other
# than this header are not kept (the justification is the prose).
# `scripts/gate_calibration_attestation.sh` validates every waiver and reports
# the records it covers as `attested_by_replay (waiver <id>)`.
"""


@dataclass
class ReplayRecord:
    record: str
    exercises: list[str]


@dataclass
class ReplayOutcome:
    replayed_at: str
    records: list[str]
    identical: bool
    differences: list[str]
    fingerprints: dict[str, str]


@dataclass
class Waiver:
    id: str
    from_: str
    to: str
    reviewed_by: str
    justification: str
    paths: list[str]
    replay: list[ReplayRecord]
    outcome: ReplayOutcome | None = None


def _strings(value: object) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def parse_waivers(text: str) -> tuple[list[Waiver], list[str]]:
    """Waivers and the problems that keep a row from being one."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [], [f"{WAIVERS.name}: not valid TOML: {exc}"]
    waivers: list[Waiver] = []
    problems: list[str] = []
    for i, row in enumerate(data.get("waiver", []), 1):
        label = f"waiver {row.get('id', f'#{i}')}"
        bad = []
        for key in ("id", "from", "to", "reviewed_by", "justification"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                bad.append(f"{label}: `{key}` must be a non-empty string")
        paths = _strings(row.get("paths"))
        if not paths:
            bad.append(f"{label}: `paths` must be a non-empty list of surface paths")
        replay = []
        rows = row.get("replay")
        if not isinstance(rows, list) or not rows:
            bad.append(f"{label}: needs at least one [[waiver.replay]] record")
            rows = []
        for j, item in enumerate(rows, 1):
            exercises = _strings(item.get("exercises")) if isinstance(item, dict) else None
            record = item.get("record") if isinstance(item, dict) else None
            if not isinstance(record, str) or not record:
                bad.append(f"{label}: replay #{j} needs a `record` id")
                continue
            if exercises is None:
                bad.append(f"{label}: replay {record}: `exercises` must be a list of paths")
                exercises = []
            replay.append(ReplayRecord(record, exercises))
        outcome = None
        raw = row.get("outcome")
        if raw is not None:
            fingerprints = raw.get("fingerprints", {}) if isinstance(raw, dict) else None
            records = _strings(raw.get("records")) if isinstance(raw, dict) else None
            differences = _strings(raw.get("differences", [])) if isinstance(raw, dict) else None
            if (
                not isinstance(raw, dict)
                or not isinstance(raw.get("replayed_at"), str)
                or not isinstance(raw.get("identical"), bool)
                or records is None
                or differences is None
                or not isinstance(fingerprints, dict)
                or not all(isinstance(v, str) for v in fingerprints.values())
            ):
                bad.append(
                    f"{label}: `outcome` needs replayed_at (string), identical (bool), "
                    "records and differences (string lists) and fingerprints (table)"
                )
            else:
                outcome = ReplayOutcome(
                    raw["replayed_at"], records, raw["identical"], differences, dict(fingerprints)
                )
        problems += bad
        if not bad:
            waivers.append(
                Waiver(
                    row["id"],
                    row["from"],
                    row["to"],
                    row["reviewed_by"],
                    row["justification"],
                    paths or [],
                    replay,
                    outcome,
                )
            )
    return waivers, problems


def load_waivers(path: Path = WAIVERS) -> tuple[list[Waiver], list[str]]:
    if not path.is_file():
        return [], []
    return parse_waivers(path.read_text())


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_list(values: list[str], indent: bool = False) -> str:
    if not indent:
        return "[" + ", ".join(_toml_str(v) for v in values) + "]"
    return "[\n" + "".join(f"  {_toml_str(v)},\n" for v in values) + "]"


def render_waivers(waivers: list[Waiver]) -> str:
    out = [WAIVERS_HEADER]
    for w in waivers:
        prose = w.justification.strip("\n").replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        out += [
            "[[waiver]]",
            f"id = {_toml_str(w.id)}",
            f"from = {_toml_str(w.from_)}",
            f"to = {_toml_str(w.to)}",
            f"reviewed_by = {_toml_str(w.reviewed_by)}",
            f'justification = """\n{prose}\\\n"""',
            f"paths = {_toml_list(w.paths, indent=True)}",
            "",
        ]
        for r in w.replay:
            out += [
                "[[waiver.replay]]",
                f"record = {_toml_str(r.record)}",
                f"exercises = {_toml_list(r.exercises, indent=True)}",
                "",
            ]
        if w.outcome is not None:
            o = w.outcome
            out += [
                "[waiver.outcome]",
                f"replayed_at = {_toml_str(o.replayed_at)}",
                f"identical = {'true' if o.identical else 'false'}",
                f"records = {_toml_list(o.records, indent=True)}",
                f"differences = {_toml_list(o.differences, indent=True)}",
                "",
                "[waiver.outcome.fingerprints]",
                *(f"{_toml_str(k)} = {_toml_str(v)}" for k, v in sorted(o.fingerprints.items())),
                "",
            ]
    return "\n".join(out)


def _canonical(value: object) -> object:
    if isinstance(value, float):
        return {"f64": value.hex()}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    return value


def fingerprint(rec: dict) -> str:
    """Digest of a stored record's emitted fields, floats by their exact bits."""
    body = {k: _canonical(v) for k, v in rec.items() if k not in NOT_EMITTED}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _bit_equal(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) or isinstance(b, float):
        return (
            isinstance(a, float)
            and isinstance(b, float)
            and struct.pack("<d", a) == struct.pack("<d", b)
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_bit_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_bit_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def compare_replay(stored: dict, payload: dict) -> list[str]:
    """Every emitted field of `stored` that the replayed payload does not reproduce
    bit for bit, named. The covered count is derived from observed × replicates."""
    rid = stored.get("id")
    differences = []

    def covered(rec: dict) -> object:
        observed, replicates = rec.get("observed"), rec.get("replicates")
        if isinstance(observed, float) and isinstance(replicates, int):
            return round(observed * replicates)
        return None

    if covered(stored) != covered(payload):
        differences.append(
            f"{rid}: covered stored {covered(stored)!r} replayed {covered(payload)!r}"
        )
    for key in sorted(set(stored) - NOT_EMITTED):
        if key not in payload:
            differences.append(f"{rid}: {key} not emitted by the replay")
        elif not _bit_equal(stored[key], payload[key]):
            differences.append(f"{rid}: {key} stored {stored[key]!r} replayed {payload[key]!r}")
    return differences


@dataclass
class WaiverCheck:
    valid: list[Waiver]
    problems: list[str]
    inert: list[str]


def validate_waivers(
    surface: Surface,
    waivers: list[Waiver],
    records: list[dict],
    repo: Repo,
    refs: References | None = None,
    require_outcome: bool = True,
) -> WaiverCheck:
    """Waivers whose scope and evidence hold; the problems of the others."""
    if not waivers:
        return WaiverCheck([], [], [])
    refs = refs or references(surface)
    deps = _crate_deps()
    by_id = {str(rec.get("id")): rec for rec in records}
    valid: list[Waiver] = []
    problems: list[str] = []
    inert: list[str] = []
    seen: set[str] = set()
    for w in waivers:
        bad: list[str] = []
        label = f"waiver {w.id}"
        if not WAIVER_ID.fullmatch(w.id):
            bad.append(f"{label}: id must match {WAIVER_ID.pattern}")
        if w.id in seen:
            bad.append(f"{label}: duplicate id")
        seen.add(w.id)
        if len(set(w.paths)) != len(w.paths):
            bad.append(f"{label}: a path is listed twice")
        for rel in w.paths:
            if surface.facet_of(rel) is None:
                bad.append(f"{label}: {rel} is not on the calibration surface")
        start, end = repo.resolve(w.from_), repo.resolve(w.to)
        if start is None:
            bad.append(f"{label}: from {w.from_!r} does not resolve to a commit in this clone")
        if end is None:
            bad.append(f"{label}: to {w.to!r} does not resolve to a commit in this clone")
        if start and end:
            if start == end:
                bad.append(f"{label}: from and to are the same commit")
            else:
                between = set(repo.changed(surface, start, end))
                for rel in w.paths:
                    if rel not in between:
                        bad.append(f"{label}: {rel} does not change between from and to")
        # Scope of the evidence.
        exercised: set[str] = set()
        replay_ids = [r.record for r in w.replay]
        if len(set(replay_ids)) != len(replay_ids):
            bad.append(f"{label}: a replay record is listed twice")
        for r in w.replay:
            if not r.exercises:
                bad.append(f"{label}: replay {r.record} exercises nothing")
            for rel in r.exercises:
                if rel not in w.paths:
                    bad.append(
                        f"{label}: replay {r.record} exercises {rel}, which the waiver "
                        "does not name"
                    )
            exercised |= set(r.exercises)
        for rel in w.paths:
            if rel not in exercised:
                bad.append(f"{label}: no replay record exercises {rel}")
        measured_at_from = [
            rec
            for rec in records
            if start and repo.resolve(str(rec.get("calibration_sha", ""))) == start
        ]
        if start and not measured_at_from and not bad:
            # Every record measured at `from` has been re-measured since: the
            # waiver can attest nothing, and its evidence has nothing to bind to.
            inert.append(w.id)
            continue
        for r in w.replay:
            rec = by_id.get(r.record)
            if rec is None:
                bad.append(f"{label}: replay record {r.record} is not in the registry")
                continue
            if start and repo.resolve(str(rec.get("calibration_sha", ""))) != start:
                bad.append(
                    f"{label}: replay record {r.record} was not measured at from "
                    f"({rec.get('calibration_sha')}), so its replay compares nothing"
                )
            carried = set(record_facets(rec, surface, refs))
            test = str(rec.get("test", "")).rsplit("::", 1)[0]
            for rel in r.exercises:
                facet = surface.facet_of(rel) or CORE
                if facet not in carried:
                    bad.append(
                        f"{label}: replay {r.record} exercises {rel} ({facet}), a facet "
                        "the record does not depend on"
                    )
                elif not _reaches(test, rel, deps):
                    bad.append(
                        f"{label}: replay {r.record} exercises {rel}, which its test "
                        f"{test} cannot reach (not in its crate dependency closure)"
                    )
        o = w.outcome
        if o is None:
            if require_outcome:
                bad.append(
                    f"{label}: no replay outcome; run "
                    f"`python3 scripts/calibration_facets.py replay --waiver {w.id}` at to"
                )
        else:
            if end and repo.resolve(o.replayed_at) != end:
                bad.append(f"{label}: replayed at {o.replayed_at}, not at to ({w.to})")
            if sorted(o.records) != sorted(replay_ids):
                bad.append(
                    f"{label}: the replay outcome covers {sorted(o.records)}, "
                    f"not {sorted(replay_ids)}"
                )
            if not o.identical:
                shown = "; ".join(o.differences[:MAX_PATHS_SHOWN]) or "no difference recorded"
                bad.append(f"{label}: the replay was not bit-identical: {shown}")
            elif o.differences:
                bad.append(f"{label}: identical = true but differences are recorded")
            for rid in replay_ids:
                rec = by_id.get(rid)
                if rec is not None and o.fingerprints.get(rid) != fingerprint(rec):
                    bad.append(
                        f"{label}: stored record {rid} differs from the one the replay "
                        "was compared with"
                    )
        problems += bad
        if not bad:
            valid.append(w)
    return WaiverCheck(valid, problems, inert)


@dataclass
class Assessment:
    sha: str
    resolved: str | None
    records: list[dict]
    facets: dict[str, set[str]]  # record id -> derived facets
    drifted: dict[str, list[str]]  # facet -> changed paths
    stale: list[dict]  # owe a re-measurement
    waived: dict[str, str] = field(default_factory=dict)  # record id -> waiver id
    waiver_problems: list[str] = field(default_factory=list)


def waiver_applies(
    waiver: Waiver,
    resolved_sha: str,
    record_facet_set: set[str],
    drifted_paths: list[str],
    surface: Surface,
    repo: Repo,
) -> bool:
    """Whether a valid waiver attests a record measured at `resolved_sha` whose
    surface drifted by `drifted_paths`. Mechanical scope, nothing else:

    * the record was measured at the waiver's `from`;
    * every drifted path in a facet the record depends on is one the waiver names;
    * no path in those facets changed between the waiver's `to` and the tree, so
      the replay at `to` ran exactly the surface the record depends on now.
    """
    if repo.resolve(waiver.from_) != resolved_sha:
        return False
    end = repo.resolve(waiver.to)
    if end is None:
        return False

    def relevant(paths: list[str]) -> set[str]:
        return {rel for rel in paths if (surface.facet_of(rel) or CORE) in record_facet_set}

    if not relevant(drifted_paths) <= set(waiver.paths):
        return False
    return not relevant(repo.changed(surface, end))


def assess(
    surface: Surface,
    records: list[dict],
    changes: dict[str, list[str]] | None = None,
    waivers: list[Waiver] | None = None,
    repo: Repo | None = None,
    registry: list[dict] | None = None,
    refs: References | None = None,
) -> list[Assessment]:
    """Drift per measured commit.

    `changes` (commit -> changed paths) replaces git for the self-test; `repo`
    replaces it entirely. Waivers default to the committed registry when git is
    read, and to none under `changes`. `registry` (default: `records`) is what
    the waivers' replay evidence is checked against.
    """
    refs = refs or references(surface)
    if repo is None:
        repo = (
            Repo()
            if changes is None
            else FakeRepo(
                {sha: sha for sha in changes}, {(sha, None): p for sha, p in changes.items()}
            )
        )
    if waivers is None:
        waivers, load_problems = load_waivers() if changes is None else ([], [])
    else:
        load_problems = []
    checked = (
        validate_waivers(
            surface, waivers, registry if registry is not None else records, repo, refs
        )
        if waivers
        else WaiverCheck([], [], [])
    )
    waiver_problems = load_problems + checked.problems
    by_sha: dict[str, list[dict]] = {}
    for rec in records:
        by_sha.setdefault(str(rec.get("calibration_sha", "")), []).append(rec)
    out = []
    for sha, recs in sorted(by_sha.items()):
        resolved = repo.resolve(sha) if sha else None
        facets = {
            str(rec.get("id")): set(record_facets(rec, surface, refs))
            for rec in recs
        }
        drifted: dict[str, list[str]] = {}
        stale: list[dict] = []
        waived: dict[str, str] = {}
        if resolved:
            paths = repo.changed(surface, resolved)
            for rel in paths:
                drifted.setdefault(surface.facet_of(rel) or CORE, []).append(rel)
            for rec in recs:
                rid = str(rec.get("id"))
                if not facets[rid] & drifted.keys():
                    continue
                for waiver in () if rid.startswith(GATE_PREFIX) else checked.valid:
                    if waiver_applies(waiver, resolved, facets[rid], paths, surface, repo):
                        waived[rid] = waiver.id
                        break
                else:
                    stale.append(rec)
        out.append(
            Assessment(sha, resolved, recs, facets, drifted, stale, waived, waiver_problems)
        )
    return out


@dataclass
class Tally:
    standing: int = 0
    by_replay: dict[str, int] = field(default_factory=dict)  # waiver id -> records
    owed: int = 0
    unverifiable: int = 0

    @property
    def replayed(self) -> int:
        return sum(self.by_replay.values())


def replay_summary(tally: Tally) -> str:
    ids = ", ".join(sorted(tally.by_replay))
    return f"attested_by_replay: {tally.replayed} record(s) under waiver(s) {ids}"


def attestation_verdict(tally: Tally, any_records: bool, invalid_waiver: bool) -> str:
    """The attestation gate's one-line verdict. Replay-attested records are accepted
    and counted by waiver; anything owed, unverifiable or an invalid waiver is not."""
    if tally.owed or tally.unverifiable or not any_records or invalid_waiver:
        return (
            f"NOT ATTESTED: {tally.owed} owed, {tally.unverifiable} unverifiable"
            + ("; a replay waiver is invalid" if invalid_waiver else "")
            + ("; the registry has no records" if not any_records else "")
        )
    return (
        f"ATTESTED: {tally.standing} record(s) attested and matching their calibration_sha"
        + (f"; {replay_summary(tally)}" if tally.by_replay else "")
        + "; no re-measurement is owed."
    )


def report(assessments: list[Assessment], surface: Surface) -> Tally:
    """Print the per-SHA, per-facet state and the counts of each status."""
    tally = Tally()
    total = sum(len(a.records) for a in assessments)
    problems = assessments[0].waiver_problems if assessments else []
    for problem in problems:
        print(f"INVALID WAIVER: {problem} (it attests nothing)")
    print(f"coverage records: {total}, measured at {len(assessments)} commit(s)")
    for a in assessments:
        print(f"  measured at {a.sha or '<no calibration_sha>'}: {len(a.records)} record(s)")
        if not a.resolved:
            tally.unverifiable += len(a.records)
            print(
                "    UNVERIFIABLE: that commit is not in this clone, so whether these records "
                "still stand cannot be decided"
            )
            continue
        for facet in sorted(surface.facets | a.drifted.keys()):
            carrying = sum(1 for ids in a.facets.values() if facet in ids)
            if facet in a.drifted:
                paths = a.drifted[facet]
                print(
                    f"    facet {facet}: DRIFTED, {len(paths)} path(s) changed; "
                    f"{carrying} record(s) depend on it"
                )
                for rel in paths[:MAX_PATHS_SHOWN]:
                    print(f"      {rel}")
                if len(paths) > MAX_PATHS_SHOWN:
                    print(f"      ... and {len(paths) - MAX_PATHS_SHOWN} more")
            elif carrying:
                print(f"    facet {facet}: matches; {carrying} record(s) depend on it")
        by_waiver: dict[str, int] = {}
        for waiver_id in a.waived.values():
            by_waiver[waiver_id] = by_waiver.get(waiver_id, 0) + 1
        for waiver_id, count in sorted(by_waiver.items()):
            print(f"    attested_by_replay (waiver {waiver_id}): {count} record(s)")
            tally.by_replay[waiver_id] = tally.by_replay.get(waiver_id, 0) + count
        if a.stale:
            print(f"    owes a re-measurement: {len(a.stale)} record(s)")
        tally.owed += len(a.stale)
        tally.standing += len(a.records) - len(a.stale) - len(a.waived)
    print(f"attested and matching: {tally.standing} record(s)")
    if tally.by_replay:
        print(
            replay_summary(tally)
            + " (their surface changed only in paths a reviewed waiver names, and its "
            "replay reproduced the stored records bit for bit)"
        )
    if tally.owed:
        print(
            f"re-measurement owed: {tally.owed} record(s), because a facet they depend on changed "
            "since they were measured"
        )
    if tally.unverifiable:
        print(f"unverifiable: {tally.unverifiable} record(s)")
    return tally


MEASURE_COMMAND = "bash scripts/measure_calibration.sh"


def attest(assessments: list[Assessment], surface: Surface, any_records: bool) -> int:
    """The attestation gate: report, then pass only when every record stands.

    A failure names what the author has to do: the drifted facets and paths behind
    the owed records and the local command that re-measures exactly those, or the
    commits a clone lacks and how to preserve them.
    """
    tally = report(assessments, surface)
    invalid = bool(assessments and assessments[0].waiver_problems)
    verdict = attestation_verdict(tally, any_records, invalid)
    print(verdict)
    if verdict.startswith("ATTESTED"):
        return 0
    if tally.owed:
        owed_facets: dict[str, set[str]] = {}
        for a in assessments:
            for rec in a.stale:
                for facet in a.facets[str(rec.get("id"))] & a.drifted.keys():
                    owed_facets.setdefault(facet, set()).update(a.drifted[facet])
        print(
            f"FAIL: {tally.owed} coverage record(s) owe a re-measurement: a facet they depend on "
            "changed since the commit they were measured at, and no valid replay waiver covers "
            "the change."
        )
        print("Drifted facets behind them, with the changed paths:")
        for facet in sorted(owed_facets):
            paths = sorted(owed_facets[facet])
            print(f"  {facet}: {len(paths)} path(s)")
            for rel in paths:
                print(f"    {rel}")
    if tally.unverifiable:
        missing = sorted(a.sha or "<no calibration_sha>" for a in assessments if not a.resolved)
        print(
            f"FAIL: {tally.unverifiable} coverage record(s) were measured at a commit this clone "
            f"does not contain: {', '.join(missing)}. A rebase, squash or deleted branch orphans "
            "the commit a record names. Push the branch or tag that preserves it, for example"
        )
        print("  git tag calibration/<name> <sha> && git push origin calibration/<name>")
        print(f"or, if the commit is gone, re-measure those records: {MEASURE_COMMAND}")
    if invalid:
        print(f"FAIL: a replay waiver above is INVALID; fix or delete it in {WAIVERS.name}.")
    if not any_records:
        print(f"FAIL: {RECORDS.relative_to(ROOT)} has no records.")
    if tally.owed:
        print("Measure on a development machine, from a clean checkout of the commit to upload:")
        print(
            f"  {MEASURE_COMMAND}            "
            "# re-measures only the owed records, collects them, re-runs this check"
        )
        print(f"  {MEASURE_COMMAND} --dry-run  # what it would run, with a rough duration")
        print(
            f"then commit {RECORDS.relative_to(ROOT)} with the files the collector regenerates, "
            "and push."
        )
    return 1


# --------------------------------------------------------------------------
# Replay: re-run a waiver's evidence records through the unchanged gate.
# --------------------------------------------------------------------------


def _gate_groups_for(records: list[dict]) -> tuple[list, int, dict[str, list]]:
    """Gate groups (scripts/calibration_groups.py owns the mapping) measuring each record."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_groups

    groups = calibration_groups.gate_groups()
    matches: dict[str, list] = {}
    for rec in records:
        found = [g for g in groups if calibration_groups.measures(g.head, g.test_filter, rec)]
        # A cargo filter matches by substring; a group naming the test exactly is
        # the one that measured it, and running the others would only cost time.
        fn = str(rec["test"]).rpartition("::")[2]
        exact = [g for g in found if g.label.partition(": ")[2] == fn]
        matches[str(rec["id"])] = exact or found
    return groups, len(groups), matches


def _safe_label(label: str) -> str:
    """The log name scripts/gate_calibration.sh derives (`tr ' /:' '___'`)."""
    return label.translate(str.maketrans(" /:", "___"))


def group_log_names(safe: str) -> list[str]:
    """The logs scripts/gate_calibration.sh writes for one group: one per
    sample-size grid point, and that point's recheck."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import collect_coverage_records

    return [
        name
        for point in range(collect_coverage_records.GRID_POINTS)
        for name in (f"{safe}.p{point}.log", f"{safe}.p{point}.recheck.log")
    ]


def replay_payloads(label_logs: list[Path]) -> dict[str, dict]:
    """Records of one group's logs, merged over their grid points exactly as
    scripts/collect_coverage_records.py merges them (a point's recheck wins)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import collect_coverage_records

    return collect_coverage_records.merged_records([p for p in label_logs if p.is_file()])


def replay(waiver_id: str, dry_run: bool) -> int:
    surface = load_surface()
    if surface.errors:
        for problem in surface.errors:
            print(f"FAIL: {problem}")
        return 1
    records = load_records()
    waivers, problems = load_waivers()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    matching = [w for w in waivers if w.id == waiver_id]
    if not matching:
        print(f"FAIL: no waiver {waiver_id} in {WAIVERS.relative_to(ROOT)}")
        return 1
    waiver = matching[0]
    repo = Repo()
    candidate = Waiver(**{**waiver.__dict__, "outcome": None})
    checked = validate_waivers(surface, [candidate], records, repo, require_outcome=False)
    if checked.problems or checked.inert:
        for problem in checked.problems:
            print(f"FAIL: {problem}")
        if checked.inert:
            print(f"FAIL: waiver {waiver_id}: no record measured at from remains to attest")
        return 1
    head, end = repo.resolve("HEAD"), repo.resolve(waiver.to)
    if head != end:
        print(f"FAIL: replay runs at to ({waiver.to} = {end}); HEAD is {head}. Check out to first.")
        return 1
    dirty = repo.changed(surface, head)
    if dirty:
        print("FAIL: the worktree's calibration surface differs from to; replay a clean checkout:")
        for rel in dirty[:MAX_PATHS_SHOWN]:
            print(f"  {rel}")
        return 1
    by_id = {str(rec["id"]): rec for rec in records}
    stored = [by_id[r.record] for r in waiver.replay]
    _, total, matches = _gate_groups_for(stored)
    plan: dict[int, object] = {}
    for rid, groups in matches.items():
        if not groups:
            print(f"FAIL: no gate group in scripts/gate_calibration.sh measures {rid}")
            return 1
        for g in groups:
            plan[g.index] = g
    print(
        f"replay of waiver {waiver_id} at {head}: {len(stored)} record(s), "
        f"{len(plan)} gate group(s)"
    )
    for g in plan.values():
        print(f"  group {g.index}{' (long)' if g.long else ''}: {g.label}")
    if dry_run:
        return 0
    out_dir = REPLAY_DIR / waiver_id
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict] = {}
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "ANTECEDENT_CALIBRATION_NSIM",
            "ANTECEDENT_CALIBRATION_RECHECK_NSIM",
            "ANTECEDENT_CALIBRATION_DRY_RUN",
            "ANTECEDENT_CALIBRATION_GRID_POINT",
            "ANTECEDENT_CALIBRATION_GRID_POINTS",
        )
    }
    for g in plan.values():
        safe = _safe_label(g.label)
        names = group_log_names(safe)
        # The gate writes into the collector's log directory; keep what the
        # measurement left there and put the replay's logs aside.
        backup = out_dir / "measurement-logs"
        backup.mkdir(exist_ok=True)
        for name in names:
            if (LOG_DIR / name).exists():
                shutil.move(str(LOG_DIR / name), str(backup / name))
            if (out_dir / name).exists():
                (out_dir / name).unlink()
        print(f"== replay group {g.index}: {g.label} ==", flush=True)
        status = subprocess.run(
            ["bash", str(ROOT / "scripts" / "gate_calibration.sh")],
            cwd=ROOT,
            env={**env, "ANTECEDENT_CALIBRATION_SHARD": f"{g.index - 1}/{total}"},
        ).returncode
        for name in names:
            if (LOG_DIR / name).exists():
                shutil.move(str(LOG_DIR / name), str(out_dir / name))
            if (backup / name).exists():
                shutil.move(str(backup / name), str(LOG_DIR / name))
        if status != 0:
            print(
                f"note: group {g.label} failed its gate at to; its records are compared as emitted"
            )
        try:
            payloads.update(replay_payloads([out_dir / n for n in names]))
        except SystemExit as refused:
            # A grid point that emitted no line (or another construction) leaves
            # its records unmerged; every stored record of the group then differs.
            print(f"note: group {g.label}: {refused}")
    differences: list[str] = []
    for rec in stored:
        payload = payloads.get(str(rec["id"]))
        if payload is None:
            differences.append(f"{rec['id']}: the replay emitted no record with this id")
        else:
            differences += compare_replay(rec, payload)
    identical = not differences
    waiver.outcome = ReplayOutcome(
        replayed_at=head,
        records=[r.record for r in waiver.replay],
        identical=identical,
        differences=differences,
        fingerprints={str(rec["id"]): fingerprint(rec) for rec in stored},
    )
    WAIVERS.write_text(render_waivers(waivers))
    for difference in differences:
        print(f"DIFFERS: {difference}")
    verdict = "bit-identical" if identical else "NOT identical: the waiver is invalid"
    print(
        f"replay of waiver {waiver_id}: {verdict}; "
        f"outcome written to {WAIVERS.relative_to(ROOT)}"
    )
    print(f"replay logs: {out_dir.relative_to(ROOT)}")
    return 0 if identical else 1


def replay_candidates(start_ref: str, end_ref: str, paths: list[str]) -> int:
    """Records measured at `from` that could serve as replay evidence for each path,
    cheapest first: short gate groups before long ones, fewer replicates first."""
    surface = load_surface()
    repo = Repo()
    start, end = repo.resolve(start_ref), repo.resolve(end_ref)
    if start is None or end is None:
        print(f"FAIL: {start_ref if start is None else end_ref} does not resolve")
        return 1
    paths = paths or repo.changed(surface, start, end)
    records = [
        rec for rec in load_records() if repo.resolve(str(rec.get("calibration_sha", ""))) == start
    ]
    if not records:
        print(f"no record in the registry was measured at {start}")
        return 1
    refs = references(surface)
    deps = _crate_deps()
    _, _, matches = _gate_groups_for(records)
    for rel in paths:
        facet = surface.facet_of(rel) or CORE
        rows = []
        for rec in records:
            carried = set(record_facets(rec, surface, refs))
            test = str(rec.get("test", "")).rsplit("::", 1)[0]
            groups = matches[str(rec["id"])]
            if facet in carried and _reaches(test, rel, deps) and groups:
                longest = any(g.long for g in groups)
                whole_file = any(": " not in g.label for g in groups)
                rows.append((longest, whole_file, rec.get("replicates", 0), str(rec["id"]), groups))
        rows.sort(key=lambda row: row[:4])
        print(f"{rel} ({facet}): {len(rows)} candidate record(s); cheapest first")
        for longest, whole_file, replicates, rid, groups in rows[:8]:
            kind = "long" if longest else ("whole-file group" if whole_file else "short")
            print(f"  {rid}  [{kind}, {replicates} replicates, group {groups[0].label}]")
    print(
        "A candidate reaches the file and carries its facet; confirm from the test that it "
        "runs the changed code before naming the file in its `exercises`."
    )
    return 0


# --------------------------------------------------------------------------
# Self-test: the guard must fail on broken input, or its pass proves nothing.
# --------------------------------------------------------------------------


def _waiver_self_test(base: Surface, refs: References, expect) -> None:
    """Replay waivers: scope, evidence and visibility, on broken inputs that must fail."""
    import contextlib
    import io
    import math

    start, end, other = "1" * 40, "2" * 40, "3" * 40
    helpers = "crates/antecedent/src/analysis/helpers.rs"  # core; every record carries it
    compile_rs = "crates/antecedent-model/src/compile.rs"  # mechanism
    stats = "crates/antecedent-stats/src/lib.rs"  # core, not waived
    temporal_adj = "crates/antecedent-estimate/src/temporal_adjustment.rs"
    measured = {
        "nominal": 0.9,
        "n_min": 400,
        "n_max": 400,
        "observed": 0.8975,
        "mcse": math.sqrt(0.8975 * 0.1025 / 400),
        "replicates": 400,
        "boundary": False,
        "role": "gated",
    }
    suite = "crates/antecedent/tests/v19_static_calibration.rs"
    cf = {
        "id": "cf",
        "test": f"{suite}::counterfactual",
        "dgp": f"{suite}::d",
        "query": "Counterfactual",
        "estimator": "gcm.fit",
        "calibration_sha": start,
        **measured,
    }
    temporal = "crates/antecedent/tests/v19_temporal_frequentist.rs"
    tp = {
        "id": "tp",
        "test": f"{temporal}::pulse",
        "dgp": f"{temporal}::d",
        "query": "PulseEffect",
        "estimator": "temporal.linear.adjustment",
        "calibration_sha": start,
        **measured,
    }
    library = "crates/antecedent-estimate/src/calibration_coverage.rs"
    lib = {
        "id": "lib",
        "test": f"{library}::linear_adjustment_analytic_ci_coverage",
        "dgp": f"{library}::d",
        "query": "Ate",
        "estimator": "linear.adjustment.ate",
        "calibration_sha": start,
        **measured,
    }
    elsewhere = tp | {"id": "tp_other", "calibration_sha": other}
    records = [cf, tp, lib, elsewhere]
    commits = {
        start: start,
        end: end,
        other: other,
        "calibration/sweep": start,
        "calibration/replay": end,
    }

    def good(**changes) -> Waiver:
        waiver = Waiver(
            id="w-good",
            from_="calibration/sweep",
            to="calibration/replay",
            reviewed_by="reviewer",
            justification='Additive fields only.\nNo interval, SE or seed path; "quoted" \\ ok.',
            paths=[compile_rs],
            replay=[ReplayRecord("cf", [compile_rs])],
            outcome=ReplayOutcome(end, ["cf"], True, [], {"cf": fingerprint(cf)}),
        )
        for key, value in changes.items():
            setattr(waiver, key, value)
        return waiver

    def diffs(after_to: list[str] | None = None, between: list[str] | None = None) -> FakeRepo:
        between_paths = [compile_rs] if between is None else between
        after = after_to or []
        return FakeRepo(
            commits,
            {
                (start, end): between_paths,
                (start, None): between_paths + after,
                (end, None): after,
                (other, None): [],
            },
        )

    def run(waiver: Waiver, repo: FakeRepo | None = None, registry: list[dict] | None = None):
        repo = repo or diffs()
        checked = validate_waivers(base, [waiver], registry or records, repo, refs)
        assessed = assess(
            base, records, waivers=[waiver], repo=repo, registry=registry, refs=refs
        )
        owed = {str(r["id"]) for a in assessed for r in a.stale}
        waived = {rid: wid for a in assessed for rid, wid in a.waived.items()}
        return checked.problems, owed, waived, assessed

    problems, owed, waived, assessed = run(good())
    expect(
        problems == [] and waived == {"cf": "w-good"} and owed == set(),
        "a valid waiver attests the records at from that depend on a named drifted facet",
    )
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        tally = report(assessed, base)
    text = printed.getvalue()
    expect(
        "attested_by_replay (waiver w-good): 1 record(s)" in text
        and "attested_by_replay: 1 record(s) under waiver(s) w-good" in text
        and "attested and matching: 3 record(s)" in text
        and tally.by_replay == {"w-good": 1}
        and tally.owed == 0,
        "the report shows replay-attested records under their own status, never as attested",
    )
    covered = Tally(standing=2, by_replay={"w-good": 3})
    expect(
        attestation_verdict(covered, True, False)
        == "ATTESTED: 2 record(s) attested and matching their calibration_sha; "
        "attested_by_replay: 3 record(s) under waiver(s) w-good; no re-measurement is owed."
        and attestation_verdict(Tally(standing=2), True, False).startswith("ATTESTED: 2 record(s)")
        and "attested_by_replay" not in attestation_verdict(Tally(standing=2), True, False)
        and attestation_verdict(covered, True, True).startswith("NOT ATTESTED")
        and attestation_verdict(Tally(by_replay={"w-good": 3}, owed=1), True, False).startswith(
            "NOT ATTESTED"
        ),
        "the attestation gate accepts attested_by_replay, printing its count and waiver ids; "
        "an invalid waiver or an owed record still fails",
    )
    _, owed, waived, _ = run(
        good(paths=[compile_rs], replay=[ReplayRecord("cf", [compile_rs])]),
        diffs(between=[compile_rs, temporal_adj]),
    )
    expect(
        "tp" in owed and set(waived) == {"cf"},
        "a changed path the waiver does not name leaves the records depending on it owing",
    )
    _, owed, waived, _ = run(good(), diffs(after_to=[compile_rs]))
    expect(
        waived == {} and owed == {"cf"},
        "a surface change outside the waiver (after to) owes the records depending on it",
    )
    _, owed, waived, _ = run(
        good(),
        FakeRepo(
            commits,
            {
                (start, end): [compile_rs],
                (start, None): [compile_rs],
                (end, None): [compile_rs],
            },
        ),
    )
    expect(
        waived == {} and owed == {"cf"},
        "a waived path changed again after to: the waiver does not apply outside its range",
    )
    expect(
        "tp_other" not in waived,
        "a record measured at a commit other than from is never covered by the waiver",
    )
    broken = good()
    broken.outcome = ReplayOutcome(
        end, ["cf"], False, ["cf: mcse stored 0.0151 replayed 0.0152"], {"cf": fingerprint(cf)}
    )
    problems, owed, waived, _ = run(broken)
    expect(
        any("not bit-identical" in p and "mcse" in p for p in problems) and waived == {},
        "a waiver whose replay recorded identical=false is invalid and names the field",
    )
    problems, _, waived, _ = run(good(replay=[ReplayRecord("cf", [])]))
    expect(
        any("exercises nothing" in p for p in problems)
        and any("no replay record exercises" in p for p in problems)
        and waived == {},
        "a replay record whose exercises is empty is invalid",
    )
    problems, _, waived, _ = run(good(replay=[ReplayRecord("cf", [compile_rs, stats])]))
    expect(
        any(f"exercises {stats}, which the waiver does not name" in p for p in problems)
        and waived == {},
        "a replay record exercising a path the waiver does not name is invalid",
    )
    problems, _, waived, _ = run(good(from_="calibration/missing"))
    expect(
        any("from 'calibration/missing' does not resolve" in p for p in problems) and waived == {},
        "a from that does not resolve is invalid",
    )
    problems, _, waived, _ = run(good(to="f" * 40))
    expect(
        any("does not resolve" in p and p.startswith("waiver w-good: to") for p in problems)
        and waived == {},
        "a to that does not resolve is invalid",
    )
    expect(Repo().resolve("0" * 40) is None, "an absent commit does not resolve in git")
    problems, _, waived, _ = run(good(), diffs(between=[helpers]))
    expect(
        any(f"{compile_rs} does not change between from and to" in p for p in problems),
        "a waived path that does not change within from..to is invalid",
    )
    problems, _, waived, _ = run(good(), registry=[cf | {"observed": 0.9}, tp, lib, elsewhere])
    expect(
        any("stored record cf differs" in p for p in problems) and waived == {},
        "a stored record that changed after the replay invalidates the waiver",
    )
    problems, _, _, _ = run(
        good(replay=[ReplayRecord("cf", [compile_rs]), ReplayRecord("lib", [compile_rs])]),
        registry=[
            cf,
            tp,
            lib | {"query": "Counterfactual", "estimator": "gcm.fit"},
            elsewhere,
        ],
    )
    expect(
        any("replay lib exercises" in p and "cannot reach" in p for p in problems),
        "a replay record whose test cannot reach the file it claims to exercise is invalid",
    )
    problems, _, _, _ = run(
        good(replay=[ReplayRecord("cf", [helpers]), ReplayRecord("tp", [compile_rs])])
    )
    expect(
        any("replay tp exercises" in p and "does not depend on" in p for p in problems),
        "a replay record exercising a facet it does not carry is invalid",
    )
    moved = good()
    moved.outcome = ReplayOutcome(other, ["cf"], True, [], {"cf": fingerprint(cf)})
    problems, _, _, _ = run(moved)
    expect(
        any("not at to" in p for p in problems),
        "a replay run at a commit other than to is invalid",
    )
    problems, _, _, _ = run(good(outcome=None))
    expect(
        any("no replay outcome" in p for p in problems),
        "a waiver without a replay outcome is invalid",
    )
    parsed, parse_problems = parse_waivers(render_waivers([good()]))
    expect(
        parse_problems == [] and len(parsed) == 1 and parsed[0] == good(),
        "the waiver registry round-trips through its writer",
    )
    _, parse_problems = parse_waivers('[[waiver]]\nid = "x"\nfrom = "a"\nto = "b"\n')
    expect(
        any("reviewed_by" in p for p in parse_problems)
        and any("paths" in p for p in parse_problems),
        "a waiver missing its review, justification or paths is rejected",
    )
    payload = {k: v for k, v in cf.items() if k not in NOT_EMITTED} | {"bound_replicates": 400}
    nudged = payload | {"mcse": math.nextafter(payload["mcse"], 1.0)}
    as_float = compare_replay(cf, payload | {"replicates": 400.0})
    missing = compare_replay(cf, {k: v for k, v in payload.items() if k != "n_max"})
    expect(
        compare_replay(cf, payload) == []
        and any(d.startswith("cf: mcse stored") for d in compare_replay(cf, nudged))
        and any(d.startswith("cf: replicates") for d in as_float)
        and any("covered" in d for d in compare_replay(cf, payload | {"observed": 0.9}))
        and any("n_max not emitted" in d for d in missing),
        "the replay comparison is bit for bit and names the differing field",
    )
    grid = [
        {"point": k, "n_min": n, "n_max": n, "observed": 0.9, "mcse": 0.015,
         "replicates": 400, "boundary": False, "role": "gated"}
        for k, n in enumerate((200, 400, 800))
    ]
    gridded = cf | {"grid": grid}
    one_point_moved = [dict(p) for p in grid]
    one_point_moved[2]["observed"] = math.nextafter(0.9, 1.0)
    expect(
        compare_replay(gridded, payload | {"grid": [dict(p) for p in grid]}) == []
        and any(
            d.startswith("cf: grid stored")
            for d in compare_replay(gridded, payload | {"grid": one_point_moved})
        )
        and fingerprint(gridded) != fingerprint(cf | {"grid": one_point_moved}),
        "a replay compares every sample-size grid point bit for bit",
    )
    committed, committed_problems = load_waivers()
    registry = load_records()
    expect(
        committed_problems == []
        and validate_waivers(base, committed, registry, Repo(), refs).problems == [],
        "the committed waiver registry is valid",
    )


def _grid_merge_self_test(expect) -> None:
    """scripts/collect_coverage_records.py merges one record's grid points, and
    refuses what would claim an unmeasured range or average a failing point."""
    import contextlib
    import io
    import tempfile

    sys.path.insert(0, str(ROOT / "scripts"))
    import collect_coverage_records as collector

    def line(point: int, n: int, observed: float, boundary: bool = False, **extra) -> str:
        payload = {
            "id": "cov.x", "query": "AverageEffect", "estimator": "aipw", "nominal": 0.95,
            "n_min": n, "n_max": n, "replicates_min": 199, "posterior_draws_min": 0,
            "unidentified_mass_max": 0.0, "observed": observed, "mcse": 0.011,
            "replicates": 400, "bound_replicates": 400, "grid_point": point,
            "boundary": boundary, "role": "gated", "test": "t", "dgp": "d",
        } | extra
        return "calibration-record " + json.dumps(payload)

    def merged(lines: dict[str, list[str]]) -> tuple[dict | None, str]:
        with tempfile.TemporaryDirectory() as tmp:
            logs = []
            for name, body in lines.items():
                path = Path(tmp) / name
                path.write_text("\n".join(body) + "\n")
                logs.append(path)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    return collector.merged_records(logs)["cov.x"], ""
            except SystemExit as refused:
                return None, str(refused)

    full = {f"g.p{k}.log": [line(k, n, obs)] for k, (n, obs) in enumerate(
        [(250, 0.945), (500, 0.948), (1000, 0.951)])}
    rec, _ = merged(full)
    expect(
        rec is not None
        and (rec["n_min"], rec["n_max"], rec["observed"]) == (250, 1000, 0.945)
        and not rec["boundary"]
        and [p["n_min"] for p in rec["grid"]] == [250, 500, 1000],
        "the collector merges a record's grid points into its measured range",
    )
    failing = dict(full) | {"g.p0.log": [line(0, 250, 0.90, True)]}
    rec, _ = merged(failing)
    expect(
        rec is not None and rec["boundary"] and rec["observed"] == 0.90,
        "a failing grid point makes the record a boundary with that point's coverage",
    )
    rechecked = dict(full) | {"g.p1.recheck.log": [line(1, 500, 0.9405, replicates=2000)]}
    rec, _ = merged(rechecked)
    expect(
        rec is not None and rec["grid"][1]["replicates"] == 2000,
        "a grid point's recheck replaces that point's first run",
    )
    _, missing = merged({k: v for k, v in full.items() if k != "g.p2.log"})
    _, flat = merged(dict(full) | {"g.p2.log": [line(2, 500, 0.95)]})
    _, other = merged(dict(full) | {"g.p2.log": [line(2, 1000, 0.95, estimator="ipw")]})
    _, smoke = merged(dict(full) | {"g.p2.log": [line(2, 1000, 0.95, smoke=True)]})
    _, pre_grid = merged({"g.log": [line(1, 500, 0.95).replace(', "grid_point": 1', "")]})
    expect(
        "not measured at grid point(s) [2]" in missing
        and "does not scale its sample size" in flat
        and "measured another construction (estimator" in other
        and "wiring smoke line" in smoke
        and "carries no grid_point" in pre_grid,
        "the collector refuses a missing point, an unscaled design, a changed "
        "construction, a smoke line and a pre-grid line",
    )


def _attestation_gate_self_test(
    base: Surface, refs: References, records: list[dict], expect
) -> None:
    """The gate every PR runs fails on a drifted record without a waiver and on a
    record whose commit does not resolve, and says what to do about each."""
    import contextlib
    import io

    def gate(recs: list[dict], changes: dict[str, list[str]] | None, repo: Repo | None = None):
        assessed = assess(base, recs, changes, waivers=[] if repo else None, repo=repo, refs=refs)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = attest(assessed, base, bool(recs))
        return status, printed.getvalue()

    status, text = gate(records, {"a" * 40: [], "b" * 40: []})
    expect(status == 0 and "ATTESTED: 3 record(s)" in text, "the gate passes an attested registry")
    compile_rs = "crates/antecedent-model/src/compile.rs"
    status, text = gate(records, {"a" * 40: [compile_rs], "b" * 40: []})
    expect(
        status == 1
        and "FAIL: 1 coverage record(s) owe a re-measurement" in text
        and f"  mechanism: 1 path(s)\n    {compile_rs}\n" in text
        and MEASURE_COMMAND in text,
        "a drifted record with no waiver fails the gate, naming the facet, its paths, the "
        "count and the local command",
    )
    status, text = gate(records, {"a" * 40: []})
    expect(
        status == 1
        and "FAIL: 2 coverage record(s) owe" not in text
        and f"measured at a commit this clone does not contain: {'b' * 40}" in text
        and "git push origin calibration/<name>" in text,
        "a record whose commit does not resolve fails the gate and says to push what preserves it",
    )
    # The same two failures through real git, as CI reads them.
    head = resolve("HEAD")
    root = _git("rev-list", "--max-parents=0", "HEAD").stdout.split()
    expect(bool(head and root), "the self-test runs in a clone with full history")
    if head and root:
        real = [dict(rec) for rec in load_records()[:1]]
        orphan = "deadbeef" * 5
        status, text = gate([real[0] | {"calibration_sha": root[-1]}], None, Repo())
        expect(
            status == 1
            and "FAIL: 1 coverage record(s) owe a re-measurement" in text
            and "Drifted facets behind them" in text,
            "through git: a record measured at a commit whose surface differs fails the gate",
        )
        status, text = gate([real[0] | {"calibration_sha": orphan}], None, Repo())
        expect(
            status == 1
            and f"does not contain: {orphan}" in text
            and "owe a re-measurement" not in text,
            "through git: a record naming a commit the clone lacks fails the gate",
        )


def self_test() -> int:
    failures: list[str] = []

    def expect(ok: bool, label: str) -> None:
        print(f"  {'ok' if ok else 'FAILED'}: {label}")
        if not ok:
            failures.append(label)

    def with_code(rel: str, extra: str, surface: Surface) -> list[str]:
        _CODE_CACHE.clear()
        _CODE_CACHE[rel] = _code(rel) + extra
        try:
            return check(surface)
        finally:
            _CODE_CACHE.clear()

    base = load_surface()
    expect(
        _test_only_module("crates/antecedent-estimate/src/calibration_coverage.rs"),
        "#[cfg(test)] plus lint allows still marks a test-only module",
    )
    defs = definitions(
        "pub struct Op;\n"
        "impl Op {\n    pub const fn query(&self) -> u8 { 0 }\n    fn new() -> Self { Op }\n}\n"
        "impl<T: Fn() -> u8> Tr for Op {\n    type Out = T;\n    fn fmt(&self) {}\n}\n"
        "pub trait Tr {\n    type Out;\n    fn fmt(&self);\n}\n"
        "pub fn compile() {}\n"
        "#[cfg(test)]\nmod tests {\n    fn context() {}\n}\n"
    )
    expect(
        defs == {"Op", "Tr", "compile"},
        "methods, associated items and inline test scaffolding are not bare-name definitions",
    )
    expect(check(base) == [], "the committed list passes")

    temporal = "crates/antecedent/src/analysis/execute/temporal_path.rs"
    problems = with_code(temporal, "\nfn probe() { let _ = counterfactual_ite; }\n", base)
    expect(
        any(p.startswith(f"{temporal} names mechanism items") for p in problems),
        "a core file calling a mechanism entry point through a glob import fails",
    )
    static = "crates/antecedent/src/analysis/execute/static_path.rs"
    problems = with_code(static, "\nuse antecedent_model::MechanismRegistry;\n", base)
    expect(
        any(p.startswith(f"{static} names mechanism items") for p in problems),
        "a core file importing from a mechanism crate fails",
    )
    result = "crates/antecedent/src/result.rs"
    problems = with_code(result, "\npub use antecedent_model::MechanismRegistry;\n", base)
    expect(
        any("re-exports mechanism items" in p for p in problems),
        "re-exporting a mechanism crate item from a core file fails",
    )
    execute = "crates/antecedent/src/analysis/execute/mod.rs"
    problems = with_code(execute, "\npub(super) use crate::gcm::MechanismRegistry;\n", base)
    expect(
        any(
            p.startswith(f"{execute} names mechanism items outside the facet: MechanismRegistry (")
            for p in problems
        ),
        "widening a reviewed re-export by one name fails",
    )
    text = LIST.read_text()
    stale = load_surface(
        text=text.replace(
            "allow mechanism crates/antecedent/src/lib.rs gcm\n",
            "allow mechanism crates/antecedent/src/lib.rs gcm fit_gcm\n",
        )
    )
    expect(
        any("stale names fit_gcm" in p for p in check(stale)),
        "an allow entry naming something the file no longer uses fails",
    )
    unmapped = load_surface(text=text.replace("core crates/antecedent-kernels/src/\n", ""))
    expect(
        any("omits crates/antecedent-kernels/src/lib.rs" in p for p in check(unmapped)),
        "a crate the list does not cover fails",
    )
    expect(
        unmapped.facet_of("crates/antecedent-kernels/src/lib.rs") is None
        and base.facet_of("crates/antecedent-kernels/src/lib.rs") == CORE
        and base.facet_of("crates/antecedent-model/src/mechanism.rs") == "mechanism"
        and base.facet_of("crates/antecedent/src/analysis/execute/attribution_path.rs")
        == "mechanism"
        and base.facet_of("crates/antecedent/src/analysis/execute/new_file.rs") == CORE,
        "longest-path facet resolution; unclaimed paths under a core line are core",
    )
    code = rust_code(
        'let s = "crates/*/src"; a(); /* x /* nested */ y */ b(); // fit_gcm\n'
        "let c = '\"'; let r = r#\"fit_gcm\"#; d::<'a>();\n"
    )
    expect(
        "fit_gcm" not in code and "a();" in code and "b();" in code and "d::<'a>()" in code,
        "comments and literals are not code; code after them survives",
    )
    tm = "fn a() { 1 }\n#[cfg(test)]\nmod tests {\n    fn t() { let s = \"}\"; /* } */ }\n}\nfn b() {}\n"
    expect(
        strip_test_modules(tm) == "fn a() { 1 }\n\nfn b() {}\n",
        "an inline test module goes, braces in its strings and comments included",
    )
    expect(
        strip_test_modules(tm.replace("fn a() { 1 }", "fn a() { 2 }")) != strip_test_modules(tm)
        and strip_test_modules(tm.replace("let s", "let z")) == strip_test_modules(tm),
        "an edit to production code counts; an edit inside the test module does not",
    )
    expect(
        strip_test_modules("#[cfg(test)]\nmod tests;\nfn a() {}\n") == "#[cfg(test)]\nmod tests;\nfn a() {}\n"
        and "cfg(test)" in strip_test_modules("#[cfg(test)]\nfn helper() {}\n"),
        "an out-of-line test module and a cfg(test) function are not stripped",
    )
    expect(
        declares_out_of_line_test_module("crates/antecedent-validate/src/tests.rs")
        and not declares_out_of_line_test_module("crates/antecedent-validate/src/validator.rs"),
        "a file declared `#[cfg(test)] mod name;` is test-only; a production file is not",
    )
    refs = references(base)
    counterfactual = {
        "test": "crates/antecedent/tests/v19_static_calibration.rs::t",
        "dgp": "crates/antecedent/tests/v19_static_calibration.rs::d",
        "query": "Counterfactual",
        "estimator": "gcm.fit",
    }
    temporal_rec = {
        "test": "crates/antecedent/tests/v19_temporal_frequentist.rs::t",
        "dgp": "crates/antecedent/tests/v19_temporal_frequentist.rs::d",
        "query": "PulseEffect",
        "estimator": "temporal.linear.adjustment",
    }
    expect(
        set(record_facets(counterfactual, base, refs))
        >= {CORE, "mechanism", "suite.v19_static_calibration"},
        "a counterfactual record carries core, mechanism and its suite",
    )
    temporal_facets = record_facets(temporal_rec, base, refs)
    expect(
        CORE in temporal_facets
        and "mechanism" not in temporal_facets
        and "suite.v19_static_calibration" not in temporal_facets
        and "estimator.temporal_adjustment" in temporal_facets
        and "identity.temporal" in temporal_facets
        and "suite.v19_temporal_frequentist" in temporal_facets,
        "a temporal record carries core, its estimator, identity and suite; not mechanism",
    )
    split = []
    for rec in load_records():
        est = str(rec.get("estimator", ""))
        if est.startswith("gcm."):
            continue
        got = [f for f in record_facets(rec, base) if f.startswith("estimator.")]
        # Empirical transport intentionally carries both the shared transport
        # surface and its empirical fitting/bootstrap surface. Keep both
        # attestation obligations; an exact-one assertion predates this route.
        if est == "transport.empirical_table_plugin":
            if set(got) != {"estimator.transport", "estimator.transport_empirical"}:
                split.append((est, got))
        elif len(got) != 1:
            split.append((est, got))
    expect(
        split == [],
        "non-mechanism estimators carry their expected estimator facets "
        "(shared and empirical facets for empirical transport)",
    )
    counterfactual |= {"id": "cf", "calibration_sha": "a" * 40}
    temporal_rec |= {"id": "tp", "calibration_sha": "a" * 40}
    older = temporal_rec | {"id": "old", "calibration_sha": "b" * 40}
    records = [counterfactual, temporal_rec, older]

    def owed(changes: dict[str, list[str]]) -> set[str]:
        return {str(r["id"]) for a in assess(base, records, changes) for r in a.stale}

    def unverifiable(changes: dict[str, list[str]]) -> set[str]:
        return {str(r["id"]) for a in assess(base, records, changes) if not a.resolved for r in a.records}

    unchanged = {"a" * 40: [], "b" * 40: []}
    expect(
        owed(unchanged) == set() and unverifiable(unchanged) == set(),
        "an unchanged surface attests every record without a re-measurement",
    )
    model = {"a" * 40: ["crates/antecedent-model/src/compile.rs"], "b" * 40: []}
    expect(
        owed(model) == {"cf"},
        "a mechanism change owes a re-measurement of the mechanism records only",
    )
    shared = {"a" * 40: [], "b" * 40: ["crates/antecedent-stats/src/lib.rs"]}
    expect(owed(shared) == {"old"}, "a core change owes every record measured at that SHA")
    for path, label in (
        ("crates/antecedent-stats/src/faer_backend.rs", "OLS backend"),
        ("crates/antecedent-prob/src/conjugate.rs", "conjugate"),
        ("crates/antecedent-kernels/src/rng.rs", "RNG"),
        ("crates/antecedent-estimate/src/util.rs", "shared estimate util"),
        ("Cargo.lock", "lockfile"),
        ("rust-toolchain.toml", "toolchain"),
    ):
        mutated = {"a" * 40: [path], "b" * 40: []}
        expect(
            owed(mutated) == {"cf", "tp"},
            f"a {label} change owes every record measured at that SHA",
        )
    suite = {"a" * 40: ["crates/antecedent/tests/v19_temporal_frequentist.rs"], "b" * 40: []}
    expect(owed(suite) == {"tp"}, "a suite change owes the records that suite emits")
    estimator = {
        "a" * 40: ["crates/antecedent-estimate/src/temporal_adjustment.rs"],
        "b" * 40: [],
    }
    expect(
        owed(estimator) == {"tp"},
        "an estimator change owes only the records that use that estimator",
    )
    other_estimator = {"a" * 40: ["crates/antecedent-estimate/src/bayesian.rs"], "b" * 40: []}
    expect(owed(other_estimator) == set(), "a different estimator's file owes nothing")
    identity = {
        "a" * 40: ["crates/antecedent-identify/src/temporal_backdoor.rs"],
        "b" * 40: [],
    }
    expect(
        owed(identity) == {"tp"},
        "an identity change owes only the records that use that identity",
    )
    other_identity = {"a" * 40: ["crates/antecedent-identify/src/backdoor.rs"], "b" * 40: []}
    expect(owed(other_identity) == set(), "a different identity's file owes nothing")
    expect(
        unverifiable({"a" * 40: []}) == {"old"},
        "a record measured at a commit missing from the clone is unverifiable",
    )
    _attestation_gate_self_test(base, refs, records, expect)
    manifest = (ROOT / "Cargo.toml").read_text()
    bumped = re.sub(r'(?m)^version = "[^"]+"', 'version = "9.9.9"', manifest)
    profile = manifest.replace('lto = "thin"', 'lto = "fat"', 1)
    expect(
        _normalized_manifest(manifest, False) == _normalized_manifest(bumped, False)
        and _normalized_manifest(manifest, False) != _normalized_manifest(profile, False),
        "a workspace version bump is not drift; a profile change is",
    )
    _waiver_self_test(base, refs, expect)
    _grid_merge_self_test(expect)
    if failures:
        print(f"calibration_facets self-test: {len(failures)} failure(s)")
        return 1
    print("calibration_facets self-test: ok")
    return 0


def widening_problems(base: str) -> list[str]:
    """How this tree's surface list narrows the list at `base`.

    Attestation compares each record against *today's* list, so re-pointing a
    file from `core` to a facet no record carries would retroactively attest
    every record against later edits to that file. A change to the list may
    therefore only widen it (add paths, or move a path toward `core`) unless the
    same change carries a replay waiver edit, which is reviewed on its own."""
    shown = _git("show", f"{base}:scripts/calibration_surface.list")
    if shown.returncode != 0:
        return [f"cannot read scripts/calibration_surface.list at {base}: {shown.stderr.strip()}"]
    old = load_surface(text=shown.stdout)
    new = load_surface()
    if _git("diff", "--quiet", base, "--", "parity/calibration_waivers.toml").returncode != 0:
        return []
    problems = []
    for facet, path in old.entries:
        now = new.facet_of(path)
        if now is None:
            problems.append(f"{path} left the calibration surface (was {facet})")
        elif now not in (facet, CORE):
            problems.append(f"{path} moved from facet {facet} to {now}, away from {CORE}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the list and the facet boundaries")
    p_wide = sub.add_parser(
        "widening", help="fail when the surface list narrows relative to a base ref"
    )
    p_wide.add_argument("base", help="ref holding the list this change is measured against")
    status = sub.add_parser("status", help="report drift since each record's SHA")
    status.add_argument("--require", action="store_true", help="fail unless every record stands")
    sub.add_parser("counts", help="records carrying each facet")
    sub.add_parser("stale-tests", help="tests whose records owe a re-measurement")
    sub.add_parser("self-test", help="the guard must fail on broken input")
    p_replay = sub.add_parser("replay", help="re-run a waiver's replay records at its `to`")
    p_replay.add_argument("--waiver", required=True)
    p_replay.add_argument("--dry-run", action="store_true", help="print the gate groups only")
    p_cand = sub.add_parser(
        "replay-candidates", help="records that could serve as a waiver's replay evidence"
    )
    p_cand.add_argument("--from", dest="start", required=True)
    p_cand.add_argument("--to", dest="end", default="HEAD")
    p_cand.add_argument("paths", nargs="*", help="default: every surface path changed from..to")
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    if args.command == "replay":
        return replay(args.waiver, args.dry_run)
    if args.command == "replay-candidates":
        return replay_candidates(args.start, args.end, args.paths)
    if args.command == "widening":
        problems = widening_problems(args.base)
        for problem in problems:
            print(f"FAIL: {problem}")
        if problems:
            print(
                "the surface list may only widen in a change without a waiver: a narrower list "
                "retroactively attests every record against later edits to the file"
            )
            return 1
        print(f"calibration surface list does not narrow relative to {args.base}")
        return 0
    surface = load_surface()
    if args.command == "check":
        problems = check(surface)
        waivers, waiver_problems = load_waivers()
        if not problems:
            checked = validate_waivers(surface, waivers, load_records(), Repo())
            waiver_problems += checked.problems
            for waiver_id in checked.inert:
                print(
                    f"note: waiver {waiver_id} is inert: every record measured at its from has "
                    "been re-measured; delete it"
                )
        for problem in problems + waiver_problems:
            print(f"FAIL: {problem}")
        if problems or waiver_problems:
            return 1
        print(
            f"calibration surface: {len(surface.entries)} paths in "
            f"{len(surface.facets)} facets; boundaries hold; "
            f"{len(waivers)} replay waiver(s) valid; list blob "
            f"{_git('hash-object', str(LIST)).stdout.strip()[:12]}"
        )
        return 0
    if surface.errors:
        for problem in surface.errors:
            print(f"FAIL: {problem}")
        return 1
    records = load_records() + load_gate_rows()
    if args.command == "counts":
        refs = references(surface)
        derived = [set(record_facets(rec, surface, refs)) for rec in records]
        for facet in sorted(surface.facets):
            carrying = sum(1 for facets in derived if facet in facets)
            print(f"{facet}: {carrying} of {len(records)} records")
        return 0
    assessments = assess(surface, records)
    if args.command == "stale-tests":
        print(
            "\n".join(
                sorted(
                    {
                        str(rec["test"])
                        for a in assessments
                        for rec in a.stale
                        if not str(rec.get("id", "")).startswith(GATE_PREFIX)
                    }
                )
            )
        )
        return 0
    if args.require:
        return attest(assessments, surface, bool(records))
    report(assessments, surface)
    return 0


if __name__ == "__main__":
    sys.exit(main())
