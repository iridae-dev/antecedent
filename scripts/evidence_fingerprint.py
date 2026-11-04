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

`scripts/scientific_records.py` records scientific passes against it and
`scripts/calibration_status.py` compares a coverage record's stored fingerprint
with it. Nothing here reads a ledger or runs a test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from _repo import cargo_target_feature_args, test_target

SKIP_DIRS = {"__pycache__", "target", ".venv", "node_modules", ".git", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".pyi", ".rs", ".toml", ".json", ".txt", ".csv", ".cfg", ".ini", ".md", ".yaml", ".yml", ".lock"}
SHARED_TEST_FILES = ("common.rs", "support.rs", "helpers.rs", "mod.rs")

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


def normalized_uv_lock(text: str) -> str:
    return re.sub(
        r'(name = "antecedent[^"]*"\n)version = "[^"]*"', r"\1version = X", text.replace("\r\n", "\n")
    )


def file_digest(path: Path) -> str:
    """Content digest of one file under the right normalization; missing is a value."""
    if not path.is_file():
        return "missing"
    data = path.read_bytes()
    if path.suffix == ".rs":
        return _sha(normalized_rust(data.decode("utf-8", "replace")))
    if path.name == "Cargo.toml" or path.name == "Cargo.lock":
        return _sha(normalized_manifest(data.decode("utf-8", "replace"), path.name.endswith(".lock")))
    if path.name == "uv.lock":
        return _sha(normalized_uv_lock(data.decode("utf-8", "replace")))
    if path.suffix in TEXT_SUFFIXES:
        return _sha(data.replace(b"\r\n", b"\n"))
    return _sha(data)


def tree_files(directory: Path, suffixes: set[str] | None = None) -> list[Path]:
    found: list[Path] = []
    if not directory.is_dir():
        return found
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(base) / name
            if suffixes is not None and path.suffix not in suffixes:
                continue
            if path.suffix in {".pyc", ".so", ".pyd", ".dylib", ".dll"}:
                continue
            found.append(path)
    return found


def tree_digest(root: Path, paths: list[Path]) -> str:
    return _sha(*[x for p in sorted(set(paths)) for x in (p.relative_to(root).as_posix(), file_digest(p))])


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


class Fingerprinter:
    """Computes assertion fingerprints for one workspace state (memoizing shared parts)."""

    def __init__(
        self,
        root: Path,
        metadata: dict | None = None,
        features: str | None = None,
        platform: bool = True,
    ):
        self.root = root
        self.platform = platform
        self.graph = workspace_graph(metadata if metadata is not None else load_metadata(root), root)
        self.features = features if features is not None else os.environ.get("ANTECEDENT_SCIENTIFIC_FEATURES", "")
        self._crate: dict[str, str] = {}
        self._shared: str | None = None

    # -- shared inputs -----------------------------------------------------
    def shared(self) -> str:
        if self._shared is None:
            r = self.root
            toolchain = [r / "rust-toolchain.toml", r / "rust-toolchain"]
            self._shared = _sha(
                file_digest(r / "Cargo.lock"),
                file_digest(r / "Cargo.toml"),
                *[file_digest(p) for p in toolchain],
                self.features,
                f"{sys.platform}-{os.uname().machine if hasattr(os, 'uname') else ''}" if self.platform else "",
            )
        return self._shared

    def crate_digest(self, name: str) -> str:
        if name not in self._crate:
            d = self.root / self.graph[name]["dir"]
            files = tree_files(d / "src") + [d / "Cargo.toml", d / "build.rs"]
            self._crate[name] = tree_digest(self.root, [f for f in files if f.is_file()])
        return self._crate[name]

    def fixtures_digest(self, fixtures: list[str]) -> str:
        files: list[Path] = []
        for rel in sorted(fixtures):
            for candidate in (self.root / rel, self.root / f"{rel}.json"):
                if candidate.is_dir():
                    files += tree_files(candidate)
                elif candidate.is_file():
                    files.append(candidate)
        return _sha(*sorted(fixtures), tree_digest(self.root, files))

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

    def parts(self, row: dict) -> dict:
        info = locate(self.root, row)
        fixtures = list(row.get("fixtures", []))
        parts = {
            "shared": self.shared(),
            "fixtures": self.fixtures_digest(fixtures),
        }
        if info["lang"] == "rust":
            crate = info["crate"]
            parts["crates"] = {c: self.crate_digest(c) for c in closure(self.graph, crate)}
            parts["test"] = tree_digest(self.root, self.rust_test_files(row["evidence_test"], crate))
            parts["features"] = " ".join(
                cargo_target_feature_args(self.root / "crates" / crate / "Cargo.toml", info["target"])
            )
        else:
            r = self.root
            test = r / row["evidence_test"]
            conftests = [
                d / "conftest.py" for d in [test.parent, *test.parent.parents] if d == r or r in d.parents
            ]
            config = [r / "python" / n for n in ("pytest.ini", "uv.lock")]
            parts["crates"] = {c: self.crate_digest(c) for c in native_crates(self.graph)}
            parts["python"] = tree_digest(r, tree_files(r / "python" / "antecedent"))
            parts["test"] = tree_digest(r, [test, *conftests, *config])
        return parts

    def test_digest(self, row: dict) -> str | None:
        """Digest of the test-side files of a Rust row (`None` for Python): what a coverage
        capture of the row is valid for."""
        info = locate(self.root, row)
        if info["lang"] != "rust":
            return None
        return tree_digest(self.root, self.rust_test_files(row["evidence_test"], info["crate"]))

    def fingerprint(self, row: dict, executed: list[str] | None = None) -> str:
        """The row's fingerprint. With `executed` (repo-relative source files a coverage capture
        saw the Rust test run) the crate closure is replaced by those files' normalized source."""
        parts = self.parts(row)
        if executed is not None and "crates" in parts and "python" not in parts:
            del parts["crates"]
            parts["mode"] = "executed"
            parts["executed"] = {rel: file_digest(self.root / rel) for rel in sorted(set(executed))}
        return _sha(json.dumps(parts, sort_keys=True))


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
