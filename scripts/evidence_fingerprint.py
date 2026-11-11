#!/usr/bin/env python3
"""The one dependency fingerprint of a test, shared by every evidence flow.

A test's fingerprint is a sha256 over everything it depends on:

* Rust test: the normalized sources (comments blanked, literals kept) and
  normalized manifests of every workspace crate in the test crate's dependency
  closure (normal, build and, for the test crate itself, dev dependencies), from
  `cargo metadata --no-deps`; the test file, its nested module directory, its
  suite aggregator and shared helper modules under `tests/`.
* Python test: `python/antecedent/**`, the native crate closure (`python/src`,
  build script, workspace crates it depends on), the test file and its conftest
  chain, pytest configuration and `python/uv.lock` with workspace version lines
  removed.
* Both: the declared fixture/oracle files, `Cargo.lock` and the workspace
  manifest with workspace version lines normalized, `rust-toolchain.toml`, the
  feature set and the platform.
* Inputs the sources reach without declaring them (a test outcome depends on all
  of these, so each is statically discovered and hashed):
  - compile-time embeds: every `include_str!`, `include_bytes!` and `include!`
    in a hashed Rust file, with a literal path (relative to the file) or a
    `concat!(env!("CARGO_MANIFEST_DIR"), "...")` path. A form that cannot be
    resolved statically (or whose target is missing or outside the repo) raises
    `UnresolvedInput`; it is never skipped;
  - runtime reads: path literals in hashed Rust sources and in Python tests and
    their local imports that resolve to an existing file or directory in the repo
    (`env!("CARGO_MANIFEST_DIR")`/`Path::new(..)` `.join(..)` runs, `"../../parity/x"`,
    `ROOT / "parity" / "x.toml"`). A directory is hashed whole. Reads assembled
    at run time from non-literal parts are listed by `Fingerprinter.dynamic_inputs`;
  - build scripts: every crate's `build.rs` and the paths it declares with
    `cargo:rerun-if-changed=`;
  - environment: the value of every `ANTECEDENT_*` variable a hashed source names
    (`env::var`, `var_os`, `option_env!`, `env!`, any `"ANTECEDENT_..."` literal,
    `os.environ`, `os.getenv`, `rerun-if-env-changed`). Other names (`HOME`,
    `CARGO_*`, `PATH`, ...) are listed in the fingerprint by name only: their values
    are host paths and per-machine settings, and hashing them would make every record
    churn on every machine and every shell.
* Derived status is not evidence. `DERIVED_STATUS_FIELDS` names, per embedded or
  read registry, the fields derived from the evidence ledgers (the attesting
  record ids, a record's stored fingerprint). Those are removed before hashing, so
  re-deriving support from the ledgers never invalidates the ledgers.

`FINGERPRINT_VERSION` is part of every fingerprint. Raising it owes every record.

`scripts/scientific_records.py` records scientific passes against it and
`scripts/calibration_status.py` compares a coverage record's stored fingerprint
with it. Nothing here reads a ledger or runs a test.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import _repo
from _repo import cargo_target_feature_args, scientific_target_feature_args, test_target

SKIP_DIRS = {"__pycache__", "target", ".venv", "node_modules", ".git", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".pyi", ".rs", ".toml", ".json", ".txt", ".csv", ".cfg", ".ini", ".md", ".yaml", ".yml", ".lock"}
SHARED_TEST_FILES = ("common.rs", "support.rs", "helpers.rs", "mod.rs")

# Salt of every fingerprint: raising it owes every stored record.
FINGERPRINT_VERSION = 3

# What a dynamic read site (a path or env name built at run time) can read, per source file.
INPUT_DECLARATIONS = "parity/fingerprint_inputs.toml"

# Registries that carry status derived from the evidence ledgers, and that source code embeds
# (`include_str!`) or reads at run time. Support status is derived from the ledgers
# (parity/scientific_records.toml, parity/coverage_records.toml), so a fingerprint that hashed
# these fields would be invalidated by re-deriving support from the ledger it guards. Each entry
# is {repo-relative path: fields dropped before hashing}; the rest of the file (canonical JSON of
# the parsed content) is evidence and stays hashed. A field path is dotted keys; `name[]` walks a
# list; `**` matches any depth. A path absent from the file is a no-op. Add the status fields of
# any new derived registry here (the cycle-guard test pins this behaviour).
DERIVED_STATUS_FIELDS: dict[str, tuple[str, ...]] = {
    # attesting_record_ids is the `current` set of calibration_status, regenerated from the ledger.
    "crates/antecedent-io-core/data/coverage_records.json": ("attesting_record_ids",),
    # Every support cell: the derived status and what evidence it lacks.
    "crates/antecedent/data/support_matrix.json": (
        "supported[].status",
        "supported[].missing_evidence",
    ),
    # Per-unit support status, derived from the ledgers by gate_support.
    "crates/antecedent-io-core/data/unit_support_status.json": (
        "graphless[].status",
        "graphless[].missing_evidence",
        "capabilities[].status",
        "capabilities[].missing_evidence",
    ),
    # The coverage registry stores each record's dependency fingerprint.
    "parity/coverage_records.toml": ("record[].fingerprint", "record[].dependency_manifest"),
    # The scientific ledger is entirely fingerprint, commit and time of the passes.
    "parity/scientific_records.toml": (
        "record[].fingerprint",
        "record[].commit",
        "record[].dirty",
        "record[].recorded_at",
        "record[].dependency_manifest",
    ),
}


class UnresolvedInput(RuntimeError):
    """A compile-time input that cannot be resolved statically. Never caught by the callers:
    fingerprinting must fail loudly rather than skip an input."""

# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------


def _sha(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8") if isinstance(part, str) else part
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


@functools.cache
def normalized_rust(text: str) -> str:
    """Comments blanked, literals kept, trailing blanks and empty lines dropped."""
    import _rust_source

    kept = _rust_source.lex(text.replace("\r\n", "\n")).kept
    return "\n".join(line.rstrip() for line in kept.split("\n") if line.strip())


def normalized_manifest(text: str, lock: bool) -> str:
    """Workspace version lines removed (the release-bump normalization)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return text
    if lock:
        packages = data.get("package", [])
        counts: dict[str, int] = {}
        for pkg in packages:
            name = str(pkg.get("name", ""))
            if not name.startswith("antecedent"):
                counts[name] = counts.get(name, 0) + 1

        def dependency(value: str) -> str:
            name = value.split(" ", 1)[0]
            # Version/source disambiguates an edge when multiple locked packages
            # share a name. Workspace version bumps alone remain irrelevant.
            return value if counts.get(name, 0) > 1 else name

        data["package"] = [
            {k: v for k, v in pkg.items() if k != "dependencies"}
            | {"dependencies": [dependency(d) for d in pkg.get("dependencies", [])]}
            for pkg in packages
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


def normalized_uv_lock(text: str) -> str:
    return re.sub(
        r'(name = "antecedent[^"]*"\n)version = "[^"]*"', r"\1version = X", text.replace("\r\n", "\n")
    )


def _drop_fields(node, segs: list[str]) -> None:
    if not segs:
        return
    seg, rest = segs[0], segs[1:]
    if seg == "**":
        _drop_fields(node, rest)
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
        for child in list(children):
            _drop_fields(child, segs)
        return
    if seg.endswith("[]"):
        child = node.get(seg[:-2]) if isinstance(node, dict) else None
        if isinstance(child, list):
            for item in child:
                _drop_fields(item, rest)
        return
    if not isinstance(node, dict):
        return
    if not rest:
        node.pop(seg, None)
    else:
        _drop_fields(node.get(seg), rest)


def evidence_only(path: Path, data: bytes, drop: tuple[str, ...]) -> str | None:
    """Canonical JSON of the parsed file with the derived fields removed; `None` if unparseable."""
    try:
        text = data.decode("utf-8")
        parsed = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError):
        return None
    for dropped in drop:
        _drop_fields(parsed, dropped.split("."))
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)


def file_digest(path: Path, root: Path | None = None) -> str:
    """Content digest of one file under the right normalization; missing is a value.

    With `root`, a registry listed in `DERIVED_STATUS_FIELDS` is hashed without its derived fields."""
    if not path.is_file():
        return "missing"
    data = path.read_bytes()
    if root is not None:
        try:
            drop = DERIVED_STATUS_FIELDS.get(path.relative_to(root).as_posix())
        except ValueError:
            drop = None
        if drop:
            canonical = evidence_only(path, data, drop)
            if canonical is not None:
                return _sha("evidence-only", canonical)
    if path.suffix == ".rs":
        return _sha(normalized_rust(data.decode("utf-8", "replace")))
    if path.name == "Cargo.toml" or path.name == "Cargo.lock":
        return _sha(normalized_manifest(data.decode("utf-8", "replace"), path.name.endswith(".lock")))
    if path.name == "uv.lock":
        return _sha(normalized_uv_lock(data.decode("utf-8", "replace")))
    if path.suffix in TEXT_SUFFIXES:
        return _sha(data.replace(b"\r\n", b"\n"))
    return _sha(data)


# Files a host file manager writes into any directory it shows. They are never data,
# git ignores them, and binding them would make a record owed whenever a directory is
# browsed.
HOST_METADATA = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def tree_files(directory: Path, suffixes: set[str] | None = None) -> list[Path]:
    found: list[Path] = []
    if not directory.is_dir():
        return found
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name in HOST_METADATA or name.startswith("._"):
                continue
            path = Path(base) / name
            if suffixes is not None and path.suffix not in suffixes:
                continue
            if path.suffix in {".pyc", ".so", ".pyd", ".dylib", ".dll"}:
                continue
            found.append(path)
    return found


def tree_digest(root: Path, paths: list[Path]) -> str:
    return _sha(*[x for p in sorted(set(paths)) for x in (p.relative_to(root).as_posix(), file_digest(p, root))])


# --------------------------------------------------------------------------
# Workspace graph (cargo metadata --no-deps)
# --------------------------------------------------------------------------


def load_metadata(root: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked", "--offline"],
        cwd=root, text=True, capture_output=True,
    )
    if out.returncode != 0:
        out = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=root, text=True, capture_output=True, check=True,
        )
    return json.loads(out.stdout)


def workspace_graph(metadata: dict, root: Path) -> dict[str, dict]:
    """name -> {dir (relative), deps, dev}: workspace crates only."""
    names = {p["name"] for p in metadata["packages"]}
    graph = {}
    for pkg in metadata["packages"]:
        deps, dev = set(), set()
        for dep in pkg.get("dependencies", []):
            if dep["name"] not in names:
                continue
            (dev if dep.get("kind") == "dev" else deps).add(dep["name"])
        manifest = Path(pkg["manifest_path"]).resolve()
        graph[pkg["name"]] = {
            "dir": manifest.parent.relative_to(root.resolve()).as_posix(),
            "deps": deps,
            "dev": dev,
        }
    return graph


def closure(graph: dict[str, dict], crate: str, with_dev: bool = True) -> list[str]:
    """The crate plus every workspace crate it reaches through normal/build deps."""
    seen: set[str] = set()
    todo = [crate]
    first = True
    while todo:
        name = todo.pop()
        if name in seen or name not in graph:
            continue
        seen.add(name)
        todo.extend(graph[name]["deps"])
        if first and with_dev:
            todo.extend(graph[name]["dev"])
        first = False
    return sorted(seen)


def native_crates(graph: dict[str, dict]) -> list[str]:
    """Closure of the crate(s) under python/; the whole workspace when none is found."""
    roots = [n for n, g in graph.items() if g["dir"] == "python" or g["dir"].startswith("python/")]
    if not roots:
        return sorted(graph)
    return sorted({c for r in roots for c in closure(graph, r, with_dev=False)})


# --------------------------------------------------------------------------
# Static discovery of inputs the sources reach without declaring them
# --------------------------------------------------------------------------


@dataclass
class Scan:
    """What a set of source files reaches: files (directories expanded), env names, markers
    (declared-but-absent paths), and reads that cannot be resolved statically."""

    files: set[Path] = field(default_factory=set)
    env: set[str] = field(default_factory=set)
    markers: set[str] = field(default_factory=set)
    dynamic: list[str] = field(default_factory=list)
    directories: set[Path] = field(default_factory=set)
    data_files: set[Path] = field(default_factory=set)
    # (crate directory, file) read only by `#[cfg(test)]` code of that crate, which
    # is compiled only when that crate's own unit tests are the target.
    test_data_files: set[tuple[Path, Path]] = field(default_factory=set)
    # (source file, character offset, file read): where each Rust data read is written.
    sites: set[tuple[Path, int, Path]] = field(default_factory=set)

    def merge(self, other: Scan) -> None:
        self.files |= other.files
        self.env |= other.env
        self.markers |= other.markers
        self.dynamic += other.dynamic
        self.directories |= other.directories
        self.data_files |= other.data_files
        self.test_data_files |= other.test_data_files
        self.sites |= other.sites


_INCLUDE = re.compile(r"\binclude(?:_str|_bytes)?\s*!\s*[(\[{]")
_MANIFEST = r'env\s*!\s*\(\s*"CARGO_MANIFEST_DIR"\s*\)'
_RUN_START = re.compile(
    rf'(?:Path\s*::\s*new|PathBuf\s*::\s*from)\s*\(\s*(?:{_MANIFEST}|"([^"\\\n]*)")\s*\)|{_MANIFEST}'
)
_JOIN = re.compile(r'\s*\.\s*(?:join|push)\s*\(\s*"([^"\\\n]*)"\s*\)')
_LITERAL_SPAN = re.compile(r'"[ \n]*"')
_ENV_READ = re.compile(
    r"(?:\benv\s*::\s*var(?:_os)?|\boption_env\s*!|\benv\s*!)\s*\(\s*"
    r'(?:"([^"\\\n]*)"|([A-Za-z_][A-Za-z_0-9:]*))'
)
_ANTECEDENT_ENV = re.compile(r"\bANTECEDENT_[A-Z0-9_]+\b")
_RERUN = re.compile(r"cargo::?rerun-if-(changed|env-changed)=(.+)", re.S)
_PATHLIKE = re.compile(r"/?[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+/?\Z")
_BARE_SEGMENTS = re.compile(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*\Z")


_CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\s*,[^\]]*\))\s*\)\s*\]")
_CFG_TEST_FILE = re.compile(r"#\s*!\s*\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\s*,[^\]]*\))\s*\)\s*\]")


def cfg_test_regions(skeleton: str) -> list[tuple[int, int]]:
    """Spans of items under `#[cfg(test)]` (or `cfg(all(test, ...))`) in a masked skeleton.

    An `any(test, ...)` or other cfg may compile in ordinary builds and is not a region.
    A file-level `#![cfg(test)]` makes the whole file a region."""
    if _CFG_TEST_FILE.search(skeleton):
        return [(0, len(skeleton))]
    regions = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for m in _CFG_TEST.finditer(skeleton):
        pos, n = m.end(), len(skeleton)
        stack: list[str] = []
        while pos < n:
            c = skeleton[pos]
            if c in pairs:
                if c == "{" and not stack:
                    depth = 0
                    for end in range(pos, n):
                        depth += (skeleton[end] == "{") - (skeleton[end] == "}")
                        if depth == 0:
                            break
                    pos = end + 1
                    break
                stack.append(pairs[c])
            elif stack and c == stack[-1]:
                stack.pop()
            elif c == ";" and not stack:
                pos += 1
                break
            pos += 1
        regions.append((m.start(), pos))
    return regions


def _str_consts(text: str) -> dict[str, str]:
    return dict(re.findall(r'\bconst\s+(\w+)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"\\\n]*)"', text))


def module_targets(path: Path, root: Path) -> list[Path]:
    """Files compiled into the target through `mod name;` and `#[path = "..."]` declarations of
    `path`. Reuses the module reader of `scripts/test_evidence.py` and adds what it leaves out: a
    `#[path]` inside an inline module (resolved against every base the compiler could use) and a
    `#[path]` that leaves the repository (raises, since it cannot be hashed)."""
    import test_evidence as te

    out = {c for c, _ in te._declared_modules(path, root)}
    masked = te.source_masked(path)
    blocks = te.module_items(masked)
    for m in te._MOD_DECL.finditer(masked.skeleton):
        declared = te._rust_path_attribute(te._attrs_before(masked.skeleton, masked.code, m.start()))
        if declared is None:
            continue
        around = sorted(
            (b for b in blocks if b.body[0] < m.start() < b.body[1]), key=lambda b: b.body[0]
        )
        base = path.parent if path.name in ("mod.rs", "lib.rs", "main.rs") or path.parent.name == "tests" else path.with_suffix("")
        candidates = [path.parent / declared]
        if around:
            candidates.append(base / declared)
            nested = base
            for block in around:
                nested = nested / block.name
                candidates.append(nested / declared)
        for cand in candidates:
            target = _bounded(root, cand)
            if target is None:
                if cand == candidates[0]:
                    raise UnresolvedInput(
                        f"{path.relative_to(root).as_posix()}: #[path = {declared!r}] leaves the repository"
                    )
                continue
            if target.is_file():
                out.add(target)
    return sorted(out)


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _split_args(text: str, start: int) -> tuple[list[str], int]:
    """Top-level comma-separated arguments of a delimited call whose body starts at `start`."""
    depth, i, cur, args, n = 1, start, start, [], len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[cur:i])
                return args, i + 1
        elif c == "," and depth == 1:
            args.append(text[cur:i])
            cur = i + 1
        i += 1
    raise UnresolvedInput("unbalanced macro arguments")


def _eval_path_expr(expr: str, crate_dir: Path) -> str:
    """A literal, `env!("CARGO_MANIFEST_DIR")` or `concat!` of those; anything else raises."""
    expr = expr.strip()
    m = re.fullmatch(r'"((?:\\.|[^"\\])*)"', expr, re.S)
    if m:
        body = m.group(1)
        if "\\" in body:
            body = body.replace("\\\\", "\0").replace('\\"', '"')
            if "\\" in body:
                raise UnresolvedInput(f"escape in path literal {expr}")
            body = body.replace("\0", "\\")
        return body
    if re.fullmatch(_MANIFEST, expr):
        return str(crate_dir)
    m = re.fullmatch(r"(?:(?:std|core)\s*::\s*)?concat\s*!\s*[(\[{](.*)[)\]}]", expr, re.S)
    if m:
        args, _ = _split_args(m.group(1) + ")", 0)
        return "".join(_eval_path_expr(a, crate_dir) for a in args if a.strip())
    raise UnresolvedInput(f"path expression {expr!r}")


def _bounded(root: Path, path: Path) -> Path | None:
    try:
        path = Path(os.path.normpath(path))
        path.relative_to(root)
    except ValueError:
        return None
    return path


def expand(path: Path) -> list[Path]:
    return tree_files(path) if path.is_dir() else [path]


def scan_rust_text(
    path: Path, text: str, root: Path, crate_dir: Path, extra_consts: dict[str, str] | None = None
) -> tuple[Scan, list[Path]]:
    """Scan one Rust file. Returns what it reaches and the included `.rs` files to scan in turn."""
    import _rust_source

    rel = path.relative_to(root).as_posix()
    views = _rust_source.lex(text.replace("\r\n", "\n"))
    kept, skeleton = views.kept, views.kept_skeleton
    out = Scan()
    nested_rs: list[Path] = []
    covered: list[tuple[int, int]] = []
    test_regions = cfg_test_regions(skeleton)

    def read_as_data(pos: int, targets) -> None:
        targets = list(targets)
        out.sites.update((path, pos, target) for target in targets)
        if any(lo <= pos < hi for lo, hi in test_regions):
            out.test_data_files.update((crate_dir, target) for target in targets)
        else:
            out.data_files.update(targets)

    def where(pos: int) -> str:
        return f"{rel}:{_line(kept, pos)}"

    # 1. Compile-time embeds. Unresolvable is fatal.
    for m in _INCLUDE.finditer(skeleton):
        args, end = _split_args(kept, m.end())
        covered.append((m.start(), end))
        if len(args) != 1:
            raise UnresolvedInput(f"{where(m.start())}: {m.group(0)} takes one path")
        if re.search(r'env\s*!\s*\(\s*"OUT_DIR"\s*\)', args[0]):
            # Generated by the crate's build script: a function of build.rs and what it declares,
            # which are hashed. Without a build script the file cannot exist.
            if not (crate_dir / "build.rs").is_file():
                raise UnresolvedInput(f"{where(m.start())}: OUT_DIR include in a crate without build.rs")
            out.markers.add(f"generated-by-build-script:{rel}:{' '.join(args[0].split())}")
            continue
        try:
            raw = _eval_path_expr(args[0], crate_dir)
        except UnresolvedInput as exc:
            raise UnresolvedInput(f"{where(m.start())}: unresolvable include ({exc})") from None
        target = _bounded(root, path.parent / raw)
        if target is None:
            raise UnresolvedInput(f"{where(m.start())}: include target {raw!r} is outside the repository")
        if not target.is_file():
            raise UnresolvedInput(f"{where(m.start())}: include target {target.relative_to(root)} is missing")
        out.files.add(target)
        if "include_str" in m.group(0) or "include_bytes" in m.group(0):
            read_as_data(m.start(), [target])
        if target.suffix == ".rs":
            nested_rs.append(target)

    # Literal spans (contents) for the scans below.
    literals = [
        (m.start(), kept[m.start() + 1 : m.end() - 1]) for m in _LITERAL_SPAN.finditer(skeleton)
    ]

    # 2. Environment.
    consts = {**(extra_consts or {}), **_str_consts(kept)}
    for m in _ENV_READ.finditer(kept):
        name, ident = m.group(1), m.group(2)
        if name is not None:
            out.env.add(name)
        elif ident.split("::")[-1] in consts:
            out.env.add(consts[ident.split("::")[-1]])
        else:
            out.dynamic.append(f"{where(m.start())}: environment read by non-literal name {ident}")
    for _, content in literals:
        out.env.update(_ANTECEDENT_ENV.findall(content))
        if path.name == "build.rs":
            r = _RERUN.match(content.strip())
            if r and "{" in r.group(2):
                out.dynamic.append(f"{rel}: build script declares {content.strip()!r}")
            elif r and r.group(1) == "env-changed":
                out.env.add(r.group(2).strip())
            elif r:
                target = _bounded(root, crate_dir / r.group(2).strip())
                if target is None:
                    out.markers.add(f"{rel}: rerun-if-changed outside repo {r.group(2).strip()}")
                elif target.exists():
                    out.files.update(expand(target))
                    out.data_files.update(expand(target))
                    if target.is_dir():
                        out.directories.add(target)
                else:
                    out.markers.add(f"missing:{target.relative_to(root).as_posix()}")

    # 3. Runtime path reads: `Path::new(env!(..)).join("a").join("b")` runs.
    def add_candidates(rel_path: str, bases: list[Path], site: int) -> bool:
        found = False
        for base in bases:
            target = _bounded(root, base / rel_path.lstrip("/"))
            if target is None or target == root or not target.exists():
                continue
            if target.is_dir() and (target in crate_dir.parents or target == crate_dir):
                continue
            if "target" in target.relative_to(root).parts[:1]:
                continue
            out.files.update(expand(target))
            read_as_data(site, expand(target))
            if target.is_dir():
                out.directories.add(target)
                # The listing is read here too: its membership is dropped with its files
                # when no executed item reads it.
                out.sites.add((path, site, target))
            found = True
        return found

    for m in _RUN_START.finditer(kept):
        if any(a <= m.start() < b for a, b in covered):
            continue
        base_lit = m.group(1)
        segs = [base_lit] if base_lit else []
        end = m.end()
        while (j := _JOIN.match(kept, end)):
            segs.append(j.group(1))
            end = j.end()
        segs = [x for x in segs if x]
        joined = "/".join(segs)
        manifest = base_lit is None
        bases = [crate_dir] if manifest else [root, crate_dir]
        if manifest and end == m.end():
            continue  # a bare env!(..): `concat!` or the leftover check below handles it
        covered.append((m.start(), end))
        if (
            segs
            and not add_candidates(joined, bases, m.start())
            and manifest
            and _bounded(root, crate_dir / joined) == root
        ):
            out.dynamic.append(
                f"{where(m.start())}: repository root joined with {joined!r}; files read beneath it by a run-time path are not tracked"
            )
    # Join runs on an unknown base (`root.join("parity").join("x")`) resolve against the root.
    cursor = 0
    while (m := _JOIN.search(kept, cursor)):
        if any(a <= m.start() < b for a, b in covered):
            cursor = m.end()
            continue
        segs, end = [m.group(1)], m.end()
        while (j := _JOIN.match(kept, end)):
            segs.append(j.group(1))
            end = j.end()
        covered.append((m.start(), end))
        joined = "/".join(x for x in segs if x)
        if joined:
            add_candidates(joined, [root], m.start())
        cursor = end

    # 4. Standalone path literals (`"../../parity/x.toml"`, `"tests/data/y.json"`).
    for at, content in literals:
        if len(content) > 240 or not _PATHLIKE.fullmatch(content):
            continue
        add_candidates(content, [crate_dir, root], at)
    # `concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/x.toml")` outside an include.
    for m in re.finditer(r"\bconcat\s*!\s*[(\[{]", skeleton):
        if any(a <= m.start() < b for a, b in covered):
            continue
        _, end = _split_args(kept, m.end())
        call = kept[m.start() : end]
        if not re.search(_MANIFEST, call):
            continue
        covered.append((m.start(), end))
        try:
            target = _bounded(root, Path(_eval_path_expr(call, crate_dir)))
        except UnresolvedInput:
            out.dynamic.append(f"{where(m.start())}: unresolvable concat! path")
            continue
        if target is not None and target != root and target.exists():
            out.files.update(expand(target))
            read_as_data(m.start(), expand(target))
            if target.is_dir():
                out.directories.add(target)
    # CARGO_MANIFEST_DIR in any other shape cannot be followed.
    for m in re.finditer(_MANIFEST, kept):
        if not any(a <= m.start() < b for a, b in covered):
            out.dynamic.append(f"{where(m.start())}: CARGO_MANIFEST_DIR path built at run time")
    return out, nested_rs


_PY_ENV = re.compile(
    r"""(?:environ\s*\[\s*|environ\s*\.\s*(?:get|pop|setdefault)\s*\(\s*|getenv\s*\(\s*)["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)
_PY_ENV_DYNAMIC = re.compile(r"""(?:environ\s*\[\s*|environ\s*\.\s*(?:get|pop)\s*\(\s*|getenv\s*\(\s*)(?!["'])\S""")


def scan_python_env(text: str) -> set[str]:
    return set(_PY_ENV.findall(text)) | set(_ANTECEDENT_ENV.findall(text))


def _const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _flatten_path(node) -> list[str | None]:
    """Parts of a path expression; `None` marks a non-literal part."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _flatten_path(node.left) + [_const_str(node.right)]
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in {"Path", "PurePath", "PurePosixPath"}:
            return [_const_str(a) for a in node.args]
        if name == "joinpath" and isinstance(func, ast.Attribute):
            return _flatten_path(func.value) + [_const_str(a) for a in node.args]
        if name == "join" and isinstance(func, ast.Attribute) and "path" in ast.unparse(func.value):
            return [_const_str(a) for a in node.args]
    return [_const_str(node)]


def scan_python_file(path: Path, text: str, root: Path) -> tuple[Scan, list[Path]]:
    """Scan one Python file. Returns what it reaches and the local modules it imports."""
    rel = path.relative_to(root).as_posix()
    out = Scan()
    out.env = scan_python_env(text)
    for m in _PY_ENV_DYNAMIC.finditer(text):
        out.dynamic.append(f"{rel}:{_line(text, m.start())}: environment read by non-literal name")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise UnresolvedInput(f"{rel}: unparsable Python ({exc.msg})") from None

    bases = [root, path.parent, root / "python"]

    def add(rel_path: str) -> None:
        for base in bases:
            target = _bounded(root, base / rel_path.lstrip("/"))
            if target is None or target == root or not target.exists():
                continue
            if target.is_dir() and (target in path.parents or target == path.parent):
                continue
            if target.is_dir() and target == root / "python":
                continue
            out.files.update(expand(target))
            out.data_files.update(expand(target))
            if target.is_dir():
                out.directories.add(target)

    nested: set[int] = set()
    modules: list[Path] = []
    path_calls = {"Path", "PurePath", "PurePosixPath", "joinpath", "join"}
    for node in ast.walk(tree):
        chain = False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and id(node) not in nested:
            cur = node.left
            while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
                nested.add(id(cur))
                cur = cur.left
            nested.add(id(cur))
            chain = True
        elif isinstance(node, ast.Call) and id(node) not in nested:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in path_calls:
                if name == "joinpath" and isinstance(func, ast.Attribute):
                    nested.add(id(func.value))
                chain = True
        if chain:
            parts = _flatten_path(node)
            run: list[str] = []
            for idx, part in enumerate([*parts, None]):
                if part is not None:
                    run.append(part)
                    continue
                if run and (len(run) >= 2 or idx - len(run) <= 1):
                    add("/".join(x for x in run if x))
                run = []
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "/" in node.value
            and len(node.value) <= 240
            and _PATHLIKE.fullmatch(node.value)
        ):
            add(node.value)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if isinstance(node, ast.Import):
                mods = [n.split(".") for n in names]
                level = 0
            else:
                base = (node.module or "").split(".") if node.module else []
                mods = [base] + [base + [n] for n in names if n != "*"]
                level = node.level
            search = (
                [path.parents[level - 1]] if level
                else [path.parent, root / "python" / "tests", root / "python", root / "scripts"]
            )
            for mod in mods:
                if not mod:
                    continue
                for d in search:
                    for cand in (d.joinpath(*mod).with_suffix(".py"), d.joinpath(*mod, "__init__.py")):
                        c = _bounded(root, cand)
                        if (
                            c is not None
                            and c.is_file()
                            and not (set(c.relative_to(root).parts) & SKIP_DIRS)
                            and not c.is_relative_to(root / "python" / "antecedent")
                        ):
                            modules.append(c)
    return out, modules


# --------------------------------------------------------------------------
# Selectors and fingerprints
# --------------------------------------------------------------------------


def lib_module(test_rel: str) -> str:
    parts = Path(test_rel).parts
    rel = list(parts[parts.index("src") + 1:])
    rel[-1] = rel[-1].removesuffix(".rs")
    if rel[-1] in ("lib", "mod"):
        rel = rel[:-1]
    return "".join(f"{p}::" for p in rel)


def locate(root: Path, row: dict) -> dict:
    """Language, selector, crate and test-side files of one scientific row."""
    test_rel, assertion = row["evidence_test"], row["evidence_assertion"]
    if test_rel.endswith(".py"):
        return {
            "lang": "python",
            "selector": f"{test_rel.removeprefix('python/')}::{assertion}",
            "crate": None,
            "target": [],
        }
    m = re.fullmatch(r"crates/([^/]+)/(tests|src)/(.+)\.rs", test_rel)
    if not m:
        raise ValueError(f"{test_rel} is not a known test layout")
    crate, kind, stem = m.groups()
    if kind == "tests":
        name, prefix = test_target(crate, stem.split("/")[0], root)
        selector = f"{crate}::{name}::{prefix}{assertion}"
        target = ["--test", name]
    else:
        selector = f"{crate}::lib::{lib_module(test_rel)}{assertion}"
        target = ["--lib"]
    return {"lang": "rust", "selector": selector, "crate": crate, "target": target}


@dataclass(eq=False)
class Usage:
    """What one row's fingerprint read: hashed files, input directories and source files read as data."""

    files: set[Path] = field(default_factory=set)
    directories: set[Path] = field(default_factory=set)
    data_files: set[Path] = field(default_factory=set)
    test_data_files: set[tuple[Path, Path]] = field(default_factory=set)
    # Sources whose `parity/fingerprint_inputs.toml` rows were consulted.
    declaration_sources: set[str] = field(default_factory=set)
    # Data files hashed as members of a shared test directory, and files a declaration names.
    # A member no source reads and no declaration names is not an input of the target.
    shared_tree_files: set[Path] = field(default_factory=set)
    declared_files: set[Path] = field(default_factory=set)

    def merge(self, other: Usage) -> None:
        self.files |= other.files
        self.directories |= other.directories
        self.data_files |= other.data_files
        self.test_data_files |= other.test_data_files
        self.declaration_sources |= other.declaration_sources
        self.shared_tree_files |= other.shared_tree_files
        self.declared_files |= other.declared_files

    def data_reads(self, test_crate: Path | None) -> set[Path]:
        """Files read as data by code compiled when `test_crate`'s unit tests (if any) are built."""
        return self.data_files | {f for crate, f in self.test_data_files if crate == test_crate}


class Fingerprinter:
    """Computes assertion fingerprints for one workspace state (memoizing shared parts)."""

    def __init__(
        self,
        root: Path,
        metadata: dict | None = None,
        features: str | None = None,
        platform: bool = True,
        environ: dict[str, str] | None = None,
        scientific_build: bool = False,
    ):
        self.root = root
        self.platform = platform
        self.scientific_build = scientific_build
        self.environ = os.environ if environ is None else environ
        self.graph = workspace_graph(metadata if metadata is not None else load_metadata(root), root)
        self.features = features if features is not None else self.environ.get("ANTECEDENT_SCIENTIFIC_FEATURES", "")
        self._scans: dict[str, Scan] = {}
        self._file_scans: dict[tuple[Path, Path], tuple[Scan, list[Path]]] = {}
        self._fd: dict[Path, str] = {}
        self._modules: dict[Path, list[Path]] = {}
        self._test_modules: dict[Path, set[Path]] = {}
        self._declarations: list[dict] | None = None
        # Memoized digests replay what they read into every open `tracking()` usage.
        self._usage: dict[str, Usage] = {}
        self._tracking: list[Usage] = []
        self._memo_values: dict[str, object] = {}

    # -- hashing with a per-instance file cache ---------------------------
    @contextmanager
    def tracking(self):
        """Collect what the fingerprints computed inside the block read, memoized parts included."""
        usage = Usage()
        self._tracking.append(usage)
        try:
            yield usage
        finally:
            self._tracking.remove(usage)

    def _use(self, usage: Usage) -> None:
        for active in self._tracking:
            active.merge(usage)

    def _use_scan(self, scan: Scan) -> Scan:
        self._use(Usage(set(), set(scan.directories), set(scan.data_files), set(scan.test_data_files)))
        return scan

    def _memo(self, key: str, compute):
        """Run `compute` once, recording what it reads so a later hit replays it."""
        if key not in self._usage:
            with self.tracking() as usage:
                value = compute()
            self._usage[key] = usage
            self._memo_values[key] = value
        self._use(self._usage[key])
        return self._memo_values[key]

    def _digest(self, path: Path) -> str:
        self._use(Usage({path}))
        if path not in self._fd:
            self._fd[path] = file_digest(path, self.root)
        return self._fd[path]

    def _tree(self, paths) -> str:
        return _sha(
            *[x for p in sorted(set(paths)) for x in (p.relative_to(self.root).as_posix(), self._digest(p))]
        )

    # -- static input discovery -------------------------------------------
    def modules_of(self, f: Path) -> list[Path]:
        if f not in self._modules:
            self._modules[f] = module_targets(f, self.root) if f.suffix == ".rs" else []
        return self._modules[f]

    def scan_rust(self, files: list[Path], crate_dir: Path, *, test_build: bool = False) -> Scan:
        """Embeds, runtime reads, build-script inputs and env names of `files`, of the Rust
        files they `include!` and of the modules they declare (`mod x;`, `#[path]`); the modules
        are inputs too. Raises `UnresolvedInput` for an embed it cannot resolve."""
        # First the file graph: modules (gated or not by `#[cfg(test)]`) and includes.
        scans: dict[Path, tuple[Scan, list[Path], list[Path]]] = {}
        edges: dict[Path, list[tuple[Path, bool]]] = {}
        todo = [f for f in files if f.is_file()]
        while todo:
            f = todo.pop()
            if f in scans:
                continue
            modules = self.modules_of(f)
            key = (f, crate_dir)
            if key not in self._file_scans:
                extra: dict[str, str] = {}
                for module in modules:
                    extra.update(_str_consts(module.read_text(encoding="utf-8", errors="replace")))
                self._file_scans[key] = scan_rust_text(
                    f, f.read_text(encoding="utf-8", errors="replace"), self.root, crate_dir, extra
                )
            scan, nested = self._file_scans[key]
            scans[f] = scan, nested, modules
            gated = {m.resolve() for m in self.test_only_modules(f)}
            edges[f] = [(n, False) for n in nested] + [(m, m.resolve() in gated) for m in modules]
            todo.extend(nested)
            todo.extend(modules)
        # A file is ordinary code when an ungated path reaches it from a file no other file
        # declares (a crate or target root); otherwise only `cfg(test)` builds compile it.
        declared = {target.resolve() for targets in edges.values() for target, _ in targets}
        ordinary: set[Path] = set()
        todo = [f for f in scans if f.resolve() not in declared]
        while todo:
            f = todo.pop()
            if f.resolve() in ordinary:
                continue
            ordinary.add(f.resolve())
            todo.extend(target for target, gated in edges.get(f, []) if not gated)
        out = Scan()
        for f, (scan, _, modules) in scans.items():
            if test_build:
                # A test target compiles `cfg(test)` code: every read is a data read.
                scan = Scan(scan.files, scan.env, scan.markers, scan.dynamic, scan.directories,
                            scan.data_files | {d for _, d in scan.test_data_files}, set())
            elif f.resolve() not in ordinary:
                # Every read of a module compiled only under `cfg(test)` is a test read.
                scan = Scan(scan.files, scan.env, scan.markers, scan.dynamic, scan.directories, set(),
                            scan.test_data_files | {(crate_dir, d) for d in scan.data_files})
            out.merge(scan)
            out.files.update(modules)
        return self._use_scan(out)

    def test_only_modules(self, f: Path) -> set[Path]:
        """Module files `f` declares under `#[cfg(test)]` (directly or in a test-only inline module)."""
        if f not in self._test_modules:
            import test_evidence as te
            self._test_modules[f] = ({m for m, gated in te._declared_modules(f, self.root) if gated}
                                     if f.suffix == ".rs" else set())
        return self._test_modules[f]

    def scan_python(self, test: Path) -> Scan:
        """Env names, repo path literals and local imports reachable from a Python test."""
        out, seen, todo = Scan(), set(), [test]
        while todo:
            f = todo.pop()
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            scan, modules = scan_python_file(f, f.read_text(encoding="utf-8", errors="replace"), self.root)
            out.merge(scan)
            out.files.update(modules)
            todo.extend(modules)
        return self._use_scan(out)

    def crate_dir(self, name: str) -> Path:
        return self.root / self.graph[name]["dir"]

    def crate_files(self, name: str) -> list[Path]:
        d = self.crate_dir(name)
        return [f for f in tree_files(d / "src") + [d / "build.rs"] if f.is_file()]

    def crate_scan(self, name: str) -> Scan:
        if name not in self._scans:
            self._scans[name] = self.scan_rust(self.crate_files(name), self.crate_dir(name))
        return self._use_scan(self._scans[name])

    def _scan_digest(self, scan: Scan) -> str:
        return _sha(self._tree(scan.files), *sorted(scan.markers))

    def env_part(self, names: set[str]) -> dict:
        """`ANTECEDENT_*` values are hashed; every other name is listed by name only, because
        its value is host state (HOME, PATH, CARGO_*) that must not churn records."""
        return {
            "hashed": {
                n: _sha(self.environ[n]) if n in self.environ else "unset"
                for n in sorted(names)
                if n.startswith("ANTECEDENT_")
            },
            "named_only": sorted(n for n in names if not n.startswith("ANTECEDENT_")),
        }

    def declarations(self) -> list[dict]:
        """Rows of `parity/fingerprint_inputs.toml`: what a dynamic read site in `source` can read."""
        if self._declarations is None:
            path = self.root / INPUT_DECLARATIONS
            rows: list[dict] = []
            if path.is_file():
                for row in _repo.load_toml(path).get("input", []):
                    if (
                        not isinstance(row.get("source"), str)
                        or not isinstance(row.get("why"), str)
                        or not row["why"].strip()
                        or not all(isinstance(x, str) for x in row.get("inputs", []))
                        or not all(isinstance(x, str) for x in row.get("env", []))
                        or ("compile_time" in row and (
                            not isinstance(row["compile_time"], list)
                            or not all(isinstance(x, str) for x in row["compile_time"])
                        ))
                    ):
                        raise UnresolvedInput(f"{INPUT_DECLARATIONS}: malformed row {row!r}")
                    rows.append(row)
            self._declarations = rows
        return self._declarations

    def declared(self, dynamic: list[str]) -> tuple[str, set[str]]:
        """Digest and env names of the declarations covering every dynamic site; a site whose
        source file has no declaration raises. Inputs may be files, directories or globs."""
        by_source: dict[str, list[dict]] = {}
        for row in self.declarations():
            by_source.setdefault(row["source"], []).append(row)
        used: list[dict] = []
        consulted = set()
        for site in sorted(set(dynamic)):
            m = re.match(r"([^:]+?)(?::(\d+))?: ", site)
            source = m.group(1) if m else site
            consulted.add(source)
            if source not in by_source:
                raise UnresolvedInput(
                    f"{site}: dynamic input is not declared; add a row for {source} to {INPUT_DECLARATIONS}"
                )
            used += [r for r in by_source[source] if r not in used]
        files: set[Path] = set()
        markers: set[str] = set()
        env: set[str] = set()
        for row in used:
            env.update(row.get("env", []))
            for pattern in row.get("inputs", []):
                hits = sorted(self.root.glob(pattern)) if any(c in pattern for c in "*?[") else [self.root / pattern]
                hits = [h for h in hits if h.exists() and _bounded(self.root, h) is not None]
                if not hits:
                    markers.add(f"missing-declared:{pattern}")
                for hit in hits:
                    files.update(expand(hit))
                    if hit.is_dir():
                        self._use(Usage(directories={hit}))
        # A record binds the rows of the sources it consulted, not the whole registry.
        self._use(Usage(declaration_sources=consulted, declared_files=set(files)))
        rows_json = json.dumps(
            [{k: row.get(k) for k in ("source", "inputs", "env", "why")} for row in used], sort_keys=True
        )
        return _sha(rows_json, self._tree(files), *sorted(markers)), env

    def dynamic_inputs(self) -> list[str]:
        """Reads the scans found but could not resolve statically (file:line: what), for audit."""
        found = {d for scan in self._scans.values() for d in scan.dynamic}
        for scan, _ in self._file_scans.values():
            found.update(scan.dynamic)
        return sorted(found)

    # -- shared inputs -----------------------------------------------------
    def shared(self) -> str:
        return self._memo("shared", self._compute_shared)

    def _compute_shared(self) -> str:
        r = self.root
        toolchain = [r / "rust-toolchain.toml", r / "rust-toolchain"]
        return _sha(
            f"fingerprint-v{FINGERPRINT_VERSION}",
            self._digest(r / "Cargo.lock"),
            self._digest(r / "Cargo.toml"),
            *[self._digest(p) for p in toolchain],
            self.features,
            f"{sys.platform}-{os.uname().machine if hasattr(os, 'uname') else ''}" if self.platform else "",
        )

    def crate_digest(self, name: str) -> str:
        def compute() -> str:
            d = self.crate_dir(name)
            files = tree_files(d / "src") + [d / "Cargo.toml", d / "build.rs"]
            # Cargo discovers build.rs automatically, including when it is added
            # without a manifest edit. Absence is therefore an input too.
            base = self._tree(files)
            return _sha(base, self._scan_digest(self.crate_scan(name)))

        return self._memo(f"crate:{name}", compute)

    def fixtures_digest(self, fixtures: list[str]) -> str:
        files: list[Path] = []
        for rel in sorted(fixtures):
            for candidate in (self.root / rel, self.root / f"{rel}.json"):
                if candidate.is_dir():
                    files += tree_files(candidate)
                elif candidate.is_file():
                    files.append(candidate)
        return _sha(*sorted(fixtures), self._tree(files))

    # -- per language ------------------------------------------------------
    def rust_test_files(self, test_rel: str, crate: str) -> list[Path]:
        r = self.root
        path = r / test_rel
        files = [path]
        files += tree_files(path.with_suffix(""), {".rs"}) if path.with_suffix("").is_dir() else []
        tests = r / "crates" / crate / "tests"
        if test_rel.startswith(f"crates/{crate}/tests/"):
            stem = Path(test_rel).parts[3].removesuffix(".rs")
            files += list(tests.glob("suite_*.rs"))
            # Shared helper modules: any subdirectory and conventional helper files.
            files += [tests / n for n in SHARED_TEST_FILES]
            files += [f for d in sorted(tests.iterdir()) if d.is_dir() and d.name != stem for f in tree_files(d)]
        return [f for f in files if f.is_file()]

    def rust_test_side(self, row: dict, crate: str) -> tuple[str, Scan]:
        """Digest of the test files plus what they embed or read, and that scan."""
        files = self.rust_test_files(row["evidence_test"], crate)
        scan = self.scan_rust([f for f in files if f.suffix == ".rs"], self.crate_dir(crate), test_build=True)
        declared, env = self.declared(scan.dynamic)
        tests = self.root / "crates" / crate / "tests"
        self._use(Usage(shared_tree_files={f for f in files if f.suffix != ".rs" and f.parent != tests}))
        scan = Scan(scan.files, scan.env | env, scan.markers, scan.dynamic, scan.directories, scan.data_files,
                    scan.test_data_files, scan.sites)
        return _sha(self._tree(files), self._scan_digest(scan), declared), scan

    def parts(self, row: dict) -> dict:
        info = locate(self.root, row)
        fixtures = list(row.get("fixtures", []))
        parts = {
            "version": FINGERPRINT_VERSION,
            "shared": self.shared(),
            "fixtures": self.fixtures_digest(fixtures),
        }
        env: set[str] = set()
        dynamic: list[str] = []
        if info["lang"] == "rust":
            crate = info["crate"]
            parts["test"], test_scan = self.rust_test_side(row, crate)
            env |= test_scan.env
            feature_args = scientific_target_feature_args if self.scientific_build else cargo_target_feature_args
            parts["features"] = " ".join(feature_args(self.root / "crates" / crate / "Cargo.toml", info["target"]))
            members = closure(self.graph, crate)
            parts["crates"] = {c: self.crate_digest(c) for c in members}
            for c in members:
                env |= self.crate_scan(c).env
                dynamic += self.crate_scan(c).dynamic
        else:
            r = self.root
            test = r / row["evidence_test"]
            conftests = [
                d / "conftest.py" for d in [test.parent, *test.parent.parents] if d == r or r in d.parents
            ]
            config = [r / "python" / n for n in ("pytest.ini", "pyproject.toml", "uv.lock")]
            members = native_crates(self.graph)
            parts["crates"] = {c: self.crate_digest(c) for c in members}
            package = tree_files(r / "python" / "antecedent")
            parts["python"] = self._tree(package)
            py_scan = self.scan_python(test)
            for conftest in conftests:
                if conftest.is_file():
                    py_scan.merge(self.scan_python(conftest))
            parts["test"] = _sha(
                self._tree([test, *conftests, *config]), self._scan_digest(py_scan)
            )
            for c in members:
                env |= self.crate_scan(c).env
                dynamic += self.crate_scan(c).dynamic
            env |= py_scan.env
            dynamic += py_scan.dynamic
            for f in package:
                if f.suffix == ".py":
                    env |= scan_python_env(f.read_text(encoding="utf-8", errors="replace"))
        parts["declared"], declared_env = self.declared(dynamic)
        parts["env"] = self.env_part(env | declared_env)
        return parts

    def fingerprint(self, row: dict) -> str:
        """The row's fingerprint."""
        return _sha(json.dumps(self.parts(row), sort_keys=True))


def coverage_row(spec: str, dgp: str = "") -> dict:
    """The fingerprint row of a coverage record's `test` (`path.rs::name`); its DGP file rides as a fixture."""
    path, _, name = spec.partition("::")
    dgp_path = dgp.partition("::")[0]
    return {
        "evidence_test": path,
        "evidence_assertion": name,
        "fixtures": [dgp_path] if dgp_path and dgp_path != path else [],
    }


def coverage_fingerprint(fp: Fingerprinter, record: dict) -> str | None:
    """The fingerprint of a coverage record's test target at the tree `fp` reads; `None` when it cannot be computed."""
    try:
        return fp.fingerprint(coverage_row(str(record.get("test", "")), str(record.get("dgp", ""))))
    except (ValueError, OSError, KeyError):
        return None
