"""Narrow an earned record's inputs to what its execution actually depended on, without
re-running anything.

A record earned before a precision rule existed binds inputs the rule would now drop or
bind more finely. Its manifest still records everything needed to apply the rule: the
commit its Nix build used, the items its run executed (item tables and selections), and
every input's digest at that commit. Refinement checks out that commit, recomputes the
reads the earn-time analysis saw there, and rewrites the manifest's inputs:

* `parity/fingerprint_inputs.toml` bound whole becomes the rows of the sources the row's
  dynamic reads consulted (`declarations`), and only those rows' globs stay;
* a data file or directory listing read only at sites outside the executed items, and not
  a fixed or declared input, is dropped;
* a Rust file bound whole by raw bytes is bound as code (`text = "rust"`), after checking
  the commit's bytes are the ones bound; one the build never compiled (outside every
  crate's `src/`, not a build script, not read as data) is dropped.

The record, its fingerprint and its Nix build are unchanged; only inputs it provably did
not depend on are removed, and finer bindings are computed from the same bytes. Nothing
runs. The result is checked like any manifest, against today's tree.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

DECLARATIONS = "parity/fingerprint_inputs.toml"
# Bound whatever the assertion runs (Snapshot's fixed inputs).
FIXED = {".cargo/config.toml", ".cargo/config", "scripts/compile_time_environment.py", "flake.lock",
         "nix/evidence-target.nix", "nix/evidence-deps.nix", "nix/evidence-common.nix",
         "nix/evidence-driver.nix", "nix/evidence-toolchain.nix", "scripts/nix_evidence_policy.py",
         "scripts/nix_evidence_deps.py", "nix/compile-time-reviewed.json", "Cargo.lock"}


def _selected_spans(tree: Path, inputs: list[dict], tables) -> tuple[dict[str, list[tuple[int, int]]], set[str]]:
    """The byte spans the executed items cover per file, and files bound whole."""
    from evidence_items import PARTS
    from rust_items import File
    spans: dict[str, list[tuple[int, int]]] = {}
    whole = {i["path"] for i in inputs if i["kind"] == "evidence_file" and i["path"].endswith(".rs")}
    for entry in inputs:
        if entry["kind"] != "rust_items":
            continue
        rel = entry["path"]
        try:
            file = File((tree / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            whole.add(rel)
            continue
        table = tables(entry["table"])
        parts: dict[str, set[str]] = {}
        for part in PARTS:
            for index in entry["select"].get(part, []):
                if index != -1:
                    parts.setdefault(table[index], set()).add(part)
        # As `select_items` decides a read is selected: inside a full item, or a head.
        for locator, chosen in parts.items():
            node = file.by_locator.get(locator)
            if node is None:
                whole.add(rel)
                continue
            spans.setdefault(rel, []).append((node.start, node.end if "full" in chosen else node.head_end))
    return spans, whole


def refine(root: Path, tree: Path, manifest: dict, row: dict, fingerprinter) -> list[dict] | None:
    """Refined inputs for `manifest`, computed in `tree` (a checkout of its build commit);
    `None` when no input changes."""
    from evidence_manifests import Checker, declarations_digest, rust_text_digest
    from evidence_items import load_inventory
    checker = Checker(root)
    inputs = manifest["inputs"]
    with fingerprinter.tracking() as used:
        fingerprinter.parts(row)
    consulted = sorted(used.declaration_sources)
    sites: dict[str, set[tuple[str, int]]] = {}
    for scan, _ in fingerprinter._file_scans.values():
        for source, position, target in scan.sites:
            try:
                sites.setdefault(Path(target).relative_to(tree).as_posix(), set()).add(
                    (Path(source).relative_to(tree).as_posix(), position))
            except ValueError:
                continue
    consumers = {Path(p).relative_to(tree).as_posix() for p in used.data_reads(None) if Path(p).is_relative_to(tree)}
    spans, whole = _selected_spans(tree, inputs, checker.items.table)
    fixtures = set()
    for rel in row.get("fixtures", []):
        fixtures |= {rel, f"{rel}.json"}
    rows = []
    if (tree / DECLARATIONS).is_file():
        import _repo
        rows = _repo.load_toml(tree / DECLARATIONS).get("input", [])
    kept_globs = {p for r in rows if r.get("source") in used.declaration_sources for p in r.get("inputs", [])}
    crate_srcs = sorted({i["path"].rsplit("/src/", 1)[0] for i in inputs if i["kind"] == "rust_items" and "/src/" in i["path"]})
    output = (manifest.get("nix_build") or {}).get("artifact", {}).get("output")
    compiled = None
    if output and (Path(output) / "compiler-inventory").is_dir():
        inventory = load_inventory(Path(output) / "compiler-inventory")
        compiled = set(inventory.by_file) | {m["file"] for m in inventory.modules.values()}

    def selected(source: str, position: int) -> bool:
        if source in whole:
            return True
        return any(lo <= position < hi for lo, hi in spans.get(source, ()))

    out, changed = [], False
    for entry in inputs:
        kind, path = entry["kind"], entry.get("path")
        if kind == "evidence_file" and path == DECLARATIONS:
            out.append(dict(kind="declarations", path=DECLARATIONS, sources=consulted,
                            digest=declarations_digest(tree, consulted), role="identity"))
            changed = True
            continue
        if kind == "glob" and path not in kept_globs:
            changed = True
            continue
        if kind in ("evidence_file", "membership") and path not in FIXED and path not in fixtures \
                and not path.endswith("Cargo.toml") and not path.endswith(".rs"):
            reads = sites.get(path)
            if reads and not any(selected(s, p) for s, p in reads):
                changed = True
                continue
        if kind == "evidence_file" and path.endswith(".rs") and entry.get("text") is None:
            bound = tree / path
            if bound.is_file() and checker_digest(bound) == entry["digest"]:
                if (compiled is not None and path not in compiled and not path.endswith("/build.rs")
                        and not any(path.startswith(f"{c}/src/") for c in crate_srcs) and path not in consumers):
                    changed = True
                    continue
                out.append({**entry, "digest": rust_text_digest(bound.read_text(encoding="utf-8", errors="replace")),
                            "text": "rust"})
                changed = True
                continue
        out.append(entry)
    return out if changed else None


def checker_digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout(root: Path, commit: str, into: Path) -> Path:
    """A detached checkout of `commit` (the bytes a record's build used)."""
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(into), commit], cwd=root, check=True,
                   capture_output=True)
    return into


def release(root: Path, path: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=root, capture_output=True)


def store(root: Path, manifest: dict) -> str:
    """Write a manifest and its input groups; its address."""
    from evidence_manifests import INPUT_GROUPS, STORE, grouped, manifest_id
    stored, groups = grouped(manifest)
    for address, data in groups.items():
        path = root / INPUT_GROUPS / f"{address}.json"
        if not path.exists():
            path.write_bytes(data)
    ref = manifest_id(stored)
    (root / STORE / f"{ref}.json").write_text(json.dumps(stored), encoding="utf-8")
    return ref


def refine_records(root: Path, records: list[tuple[dict, dict, dict]]) -> dict[str, str]:
    """Refine each (record, manifest, row); `old ref -> new ref` for those that changed.

    Records are grouped by the commit their build used; each commit is checked out once."""
    from evidence_fingerprint import Fingerprinter
    by_commit: dict[str, list] = {}
    for record, manifest, row in records:
        commit = (manifest.get("nix_build") or {}).get("source", {}).get("commit")
        if commit:
            by_commit.setdefault(commit, []).append((record, manifest, row))
    moved: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="antecedent-refine-") as tmp:
        for n, (commit, items) in enumerate(sorted(by_commit.items())):
            tree = checkout(root, commit, Path(tmp).resolve() / f"c{n}")
            try:
                fp = Fingerprinter(tree, platform=False, scientific_build=True)
                for record, manifest, row in items:
                    inputs = refine(root, tree.resolve(), manifest, row, fp)
                    if inputs is None:
                        continue
                    new = {k: v for k, v in manifest.items() if k != "inputs"}
                    new["inputs"] = inputs
                    moved[record["dependency_manifest"]] = store(root, new)
            finally:
                release(root, tree)
    return moved
