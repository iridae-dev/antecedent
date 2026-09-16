#!/usr/bin/env python3
"""Which coverage records a change to the statistical surface invalidates.

A coverage record in `parity/coverage_records.toml` stands until the code it
measured changes. `scripts/calibration_surface.list` is the only owner of that
surface. It assigns every path to a facet:

* `core` — code every record depends on. A change here invalidates every
  record. A path under the surface that no narrower line claims is `core`, and
  a crate, manifest or record-emitting suite the list does not cover at all
  fails `check`, so an unmapped edit can never look harmless.
* any other facet — code only some records depend on. A change here
  invalidates the records that carry the facet and no others.

A record's facets are derived from the record itself, never declared by hand:
`core`; the facet of the file its `test` and `dgp` name; every facet a `key`
line in the list assigns to one of its key fields; and every non-core facet
whose items that suite file (or the shared test harness) names.

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

Usage:

    python3 scripts/calibration_facets.py check            # list + boundaries
    python3 scripts/calibration_facets.py status           # drift per measured SHA
    python3 scripts/calibration_facets.py status --require # fail if any record owes a re-measurement
    python3 scripts/calibration_facets.py counts           # records carrying each facet
    python3 scripts/calibration_facets.py stale-tests      # tests whose records owe a re-measurement
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "scripts" / "calibration_surface.list"
RECORDS = ROOT / "parity" / "coverage_records.toml"

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


_PUB_USE = re.compile(r"^[ \t]*pub(?:\([^)]*\))?[ \t]+use\b([^;]*);", re.M)


def definitions(code: str) -> set[str]:
    """Items a file defines, plus every name it re-exports with `pub use`."""
    names = {a or b for a, b in _DEF.findall(code)}
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
            rf"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+{re.escape(path.stem)}\s*;",
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
            problems.append(f"{rel} must be {CORE}: every record depends on it")
    # Boundaries.
    refs = references(surface)
    for facet, rel in refs.reexports:
        problems.append(
            f"{rel} re-exports {facet} items from outside the facet; make it a {facet} "
            "file or stop re-exporting"
        )
    for facet, hits in sorted(refs.by_facet.items()):
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
    return problems


# --------------------------------------------------------------------------
# Records.
# --------------------------------------------------------------------------


def load_records(path: Path = RECORDS) -> list[dict]:
    if not path.is_file():
        return []
    return tomllib.loads(path.read_text()).get("record", [])


def record_facets(rec: dict, surface: Surface, refs: References | None = None) -> list[str]:
    """Facets a record depends on, derived from its own fields."""
    refs = refs or references(surface)
    facets = {CORE}
    consumers = []
    for spec in (str(rec.get("test", "")), str(rec.get("dgp", ""))):
        rel = spec.rsplit("::", 1)[0]
        facets.add(surface.facet_of(rel) or CORE)
        consumers.append(rel)
    for facet, hits in refs.by_facet.items():
        if any(rel in hits for rel in consumers + refs.harness):
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


def changed_paths(surface: Surface, sha: str) -> list[str]:
    """Surface paths whose content differs between `sha` and the working tree."""
    diff = _git("diff", "--name-only", sha, "--", *surface.paths)
    if diff.returncode != 0:
        raise SystemExit(f"git diff {sha} failed: {diff.stderr.strip()}")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *surface.paths)
    changed = sorted(set(diff.stdout.split()) | set(untracked.stdout.split()))
    kept = []
    for rel in changed:
        if NORMALIZED.search(rel) and (ROOT / rel).is_file():
            old = _git("show", f"{sha}:{rel}")
            if old.returncode == 0:
                lock = rel.endswith(".lock")
                try:
                    same = _normalized_manifest(old.stdout, lock) == _normalized_manifest(
                        (ROOT / rel).read_text(), lock
                    )
                except tomllib.TOMLDecodeError:
                    same = False
                if same:
                    continue
        kept.append(rel)
    return kept


@dataclass
class Assessment:
    sha: str
    resolved: str | None
    records: list[dict]
    facets: dict[str, set[str]]  # record id -> derived facets (plus any stored ones)
    drifted: dict[str, list[str]]  # facet -> changed paths
    stale: list[dict]


def assess(
    surface: Surface, records: list[dict], changes: dict[str, list[str]] | None = None
) -> list[Assessment]:
    """Drift per measured commit. `changes` (commit -> changed paths) replaces git."""
    refs = references(surface)
    by_sha: dict[str, list[dict]] = {}
    for rec in records:
        by_sha.setdefault(str(rec.get("calibration_sha", "")), []).append(rec)
    out = []
    for sha, recs in sorted(by_sha.items()):
        if changes is None:
            resolved = resolve(sha) if sha else None
        else:
            resolved = sha if sha in changes else None
        facets = {
            str(rec.get("id")): set(record_facets(rec, surface, refs)) | set(rec.get("facets", []))
            for rec in recs
        }
        drifted: dict[str, list[str]] = {}
        stale: list[dict] = []
        if resolved:
            paths = changed_paths(surface, resolved) if changes is None else changes[sha]
            for rel in paths:
                drifted.setdefault(surface.facet_of(rel) or CORE, []).append(rel)
            stale = [rec for rec in recs if facets[str(rec.get("id"))] & drifted.keys()]
        out.append(Assessment(sha, resolved, recs, facets, drifted, stale))
    return out


def report(assessments: list[Assessment], surface: Surface) -> tuple[int, int, int]:
    """Print the per-SHA, per-facet state; return (standing, owed, unverifiable)."""
    standing = owed = unverifiable = 0
    total = sum(len(a.records) for a in assessments)
    print(f"coverage records: {total}, measured at {len(assessments)} commit(s)")
    for a in assessments:
        print(f"  measured at {a.sha or '<no calibration_sha>'}: {len(a.records)} record(s)")
        if not a.resolved:
            unverifiable += len(a.records)
            print(
                "    UNVERIFIABLE: that commit is not in this clone (fetch full history); "
                "whether these records still stand cannot be decided"
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
        owed += len(a.stale)
        standing += len(a.records) - len(a.stale)
    print(f"attested and matching: {standing} record(s)")
    if owed:
        print(
            f"re-measurement owed: {owed} record(s), because a facet they depend on changed "
            "since they were measured (`python3 scripts/calibration_facets.py stale-tests` "
            "lists their tests)"
        )
    if unverifiable:
        print(f"unverifiable: {unverifiable} record(s)")
    return standing, owed, unverifiable


# --------------------------------------------------------------------------
# Self-test: the guard must fail on broken input, or its pass proves nothing.
# --------------------------------------------------------------------------


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
        "mechanism" not in temporal_facets
        and "suite.v19_static_calibration" not in temporal_facets
        and CORE in temporal_facets,
        "a temporal record carries core but neither mechanism nor another suite",
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
    expect(owed(shared) == {"old"}, "a core change owes every record measured before it")
    suite = {"a" * 40: ["crates/antecedent/tests/v19_temporal_frequentist.rs"], "b" * 40: []}
    expect(owed(suite) == {"tp"}, "a suite change owes the records that suite emits")
    expect(
        unverifiable({"a" * 40: []}) == {"old"},
        "a record measured at a commit missing from the clone is unverifiable",
    )
    manifest = (ROOT / "Cargo.toml").read_text()
    bumped = re.sub(r'(?m)^version = "[^"]+"', 'version = "9.9.9"', manifest)
    profile = manifest.replace('lto = "thin"', 'lto = "fat"', 1)
    expect(
        _normalized_manifest(manifest, False) == _normalized_manifest(bumped, False)
        and _normalized_manifest(manifest, False) != _normalized_manifest(profile, False),
        "a workspace version bump is not drift; a profile change is",
    )
    if failures:
        print(f"calibration_facets self-test: {len(failures)} failure(s)")
        return 1
    print("calibration_facets self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the list and the facet boundaries")
    status = sub.add_parser("status", help="report drift since each record's SHA")
    status.add_argument("--require", action="store_true", help="fail unless every record stands")
    sub.add_parser("counts", help="records carrying each facet")
    sub.add_parser("stale-tests", help="tests whose records owe a re-measurement")
    sub.add_parser("self-test", help="the guard must fail on broken input")
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    surface = load_surface()
    if args.command == "check":
        problems = check(surface)
        for problem in problems:
            print(f"FAIL: {problem}")
        if problems:
            return 1
        print(
            f"calibration surface: {len(surface.entries)} paths in "
            f"{len(surface.facets)} facets; boundaries hold"
        )
        return 0
    if surface.errors:
        for problem in surface.errors:
            print(f"FAIL: {problem}")
        return 1
    records = load_records()
    if args.command == "counts":
        refs = references(surface)
        derived = [set(record_facets(rec, surface, refs)) for rec in records]
        for facet in sorted(surface.facets):
            carrying = sum(1 for facets in derived if facet in facets)
            print(f"{facet}: {carrying} of {len(records)} records")
        return 0
    assessments = assess(surface, records)
    if args.command == "stale-tests":
        print("\n".join(sorted({str(rec["test"]) for a in assessments for rec in a.stale})))
        return 0
    _, owed, unverifiable = report(assessments, surface)
    if args.require and (owed or unverifiable or not records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
