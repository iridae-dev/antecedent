#!/usr/bin/env python3
"""Rebind earned records to what their measurement can depend on, without re-measuring.

    python3 scripts/narrow_evidence_binding.py                 # dry run: counts only
    python3 scripts/narrow_evidence_binding.py --apply         # rewrite manifests and registries
    python3 scripts/narrow_evidence_binding.py --logs DIR ...  # also cross-check retained runs
    python3 scripts/narrow_evidence_binding.py --stage DIR     # dry run, keeping its manifests
    python3 scripts/narrow_evidence_binding.py --stage DIR --apply   # rebind from that dry run

A record's manifest lists the inputs its measurement was bound to. Three of them bind
more than the measurement can depend on:

* a calibration measurement binds the whole body of every function its executed code
  names, transitively, through every dispatcher (`evidence_reach` says which of those
  it can enter);
* a file an `include!` compiles in place, and the module file that includes it, were
  bound whole;
* build recipes and policy code were bound by their bytes (`evidence_recipes`).

This tool writes, for each record, a manifest with the same record, measurement and
build, whose inputs are narrowed. Nothing runs and no measured value is read or
written: the registries change only in the manifest address each record cites.

Why the narrowed manifest attests nothing the measurement did not see, checked here
for every record before anything is written:

1. The record is current: every input of its manifest still matches (a record that is
   not current is refused and left as it is).
2. What the measurement can enter is decided on the code that was measured: the
   source tree and compiler inventory of the record's own build, the one its manifest
   names. The manifest names items by locator, so every bound item must be in that
   source under the same name: the source holds each bound part and scope exactly as
   bound, or (a record rebound after an edit) each bound item is there with as many
   same-named siblings.
3. Every item part of the new selection was bound by the old manifest: selected by the
   old entry of its file, or inside a file the old manifest bound whole. The guards on
   how names and methods resolve are the narrowed selection's own, as earning writes
   them; a file that was bound whole had none, and its items now have theirs.
4. Every selected part is bound with the tokens the old manifest bound, which are the
   tokens the checkout holds, the record being current; a recipe bound by content
   hashes, in the checkout, to the bytes bound. Where a kept item reads differently in
   the build's source, an earlier rebinding judged that edit out of the record's
   reach: the item stays bound as that rebinding left it, and the dry run counts the
   records this concerns.
5. The new manifest is checked like any other and must be current.

With `--logs`, each record whose measurement logs are still held is also compared with
what its run executed: every executed function the record binds must be one the
analysis enters, in a region it left open.

A record whose build output is not in this machine's store, whose bound names do not
mean the same items in the build's source, or whose retained run executed code the
analysis closed, keeps its Rust selection as it is (its recipes are still rebound by
content). The dry run says how many records each of these left broad.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import _repo
import evidence_items
import evidence_manifests as manifests
import evidence_reach
import evidence_recipes
from evidence_items import PARTS, Names, Selection, items_digest, parse, sha
from evidence_manifests import ANALYZER, Checker, canonical, digest, grouped, manifest_id
from rust_items import ParseError, scope_entries

ROOT = Path(__file__).resolve().parents[1]
REGISTRIES = {"calibration": ("parity/coverage_records.toml", "record"),
              "scientific": ("parity/scientific_records.toml", "record"),
              "probe": ("parity/calibration_probe_attestations.toml", "attestation")}
# Whole-file reasons that mean "compiled in place", not "read as data".
_COMPILED_WHOLE = ("no compiler module for this file", "module declaration of ")


class Kept(Exception):
    """The record's Rust selection stays as it is, for the reason given."""


# ---------------------------------------------------------------------- build context

class Build:
    """One evidence build: its source tree, compiler inventory and item graph."""

    def __init__(self, output: Path, source: Path):
        self.source = source
        self.inventory = evidence_items.load_inventory(output / "compiler-inventory")
        if not self.inventory.items:
            raise Kept("the build output holds no compiler inventory")
        self._files: dict[str, object] = {}
        self.graph = evidence_reach.Graph(self.file, self.inventory)
        self.memo: dict[tuple, object] = {}

    def file(self, rel: str):
        """A parsed file of the build's source (a store path: it does not change)."""
        if rel not in self._files:
            try:
                self._files[rel] = parse(self.source / rel)
            except (ParseError, OSError):
                self._files[rel] = None
        return self._files[rel]


_BUILDS: dict[str, Build | Kept] = {}


def build_of(manifest: dict) -> Build:
    receipt = manifest.get("nix_build") or {}
    output = (receipt.get("artifact") or {}).get("output")
    source = (receipt.get("source") or {}).get("store_path")
    if not isinstance(output, str) or not isinstance(source, str):
        raise Kept("the manifest names no build output")
    if output not in _BUILDS:
        if len(_BUILDS) >= 2:  # an inventory is large: keep the two most recent
            _BUILDS.pop(next(iter(_BUILDS)))
        try:
            if not (Path(output) / "compiler-inventory").is_dir() or not Path(source).is_dir():
                raise Kept("the build output is not in this machine's store")
            if (Path(output) / "source-root").read_text(encoding="utf-8").strip() != source:
                raise Kept("the build output was not built from the source the manifest names")
            _BUILDS[output] = Build(Path(output), Path(source))
        except Kept as kept:
            _BUILDS[output] = kept
    found = _BUILDS[output]
    if isinstance(found, Kept):
        raise found
    return found


# ---------------------------------------------------------------------- the old binding

class Bound:
    """What a manifest binds of the Rust sources: item parts per file, and whole files."""

    def __init__(self, checker: Checker, manifest: dict):
        self.entries: dict[str, dict] = {}
        self.tables: dict[str, list[str]] = {}
        self.parts: dict[str, dict[str, set[str]]] = {}
        self.whole: dict[str, dict] = {}
        self.names = self.guards = None
        for entry in manifest["inputs"]:
            kind = entry["kind"]
            if kind == "rust_items":
                rel = entry["path"]
                table = checker.items.table(entry["table"])
                self.entries[rel], self.tables[rel] = entry, table
                parts = self.parts.setdefault(rel, {})
                for part in PARTS:
                    for index in entry["select"].get(part, []):
                        parts.setdefault("" if index == -1 else table[index], set()).add(part)
            elif kind == "evidence_file" and entry["path"].endswith(".rs"):
                self.whole[entry["path"]] = entry
            elif kind == "rust_names":
                self.names = entry
            elif kind == "rust_guards":
                self.guards = entry
        reasons = {}
        for line in manifest.get("whole_sources", []):
            rel, _, reason = line.partition(": ")
            reasons[rel] = reason
        # Whole files the build compiled in place: their items may be selected.
        self.compiled = {rel for rel in self.whole if reasons.get(rel, "").startswith(_COMPILED_WHOLE)}
        self.reasons = reasons

    def allows(self, rel: str, locator: str, wanted: str) -> str | None:
        """The largest part of an item, up to `wanted`, the manifest already binds."""
        if rel in self.compiled:
            return wanted
        have = self.parts.get(rel, {}).get(locator)
        if not have:
            return None
        if wanted in have or ("full" in have and wanted != "inner"):
            return wanted
        if wanted == "full":
            return next((part for part in ("sig", "head") if part in have), None)
        return None

    def count(self, build: Build | None) -> int:
        """Bound item parts; a file bound whole counts every item it holds."""
        total = sum(len(indices) for entry in self.entries.values() for indices in entry["select"].values())
        for rel in self.whole:
            file = build.file(rel) if build is not None else None
            total += len(file.by_locator) if file is not None else 1
        return total


# ---------------------------------------------------------------------- narrowing

def _roots(build: Build, bound: Bound, test: str) -> tuple[set, set]:
    """Where a measurement whose executed regions are gone can start: every bound function
    (or macro invocation) no bound item names, its test, and the impls of traits the
    workspace does not define; and the macro invocations that may write such impls."""
    graph = build.graph
    keys = {(rel, locator) for rel, parts in bound.parts.items() for locator in parts if locator}
    for rel in bound.compiled:
        file = build.file(rel)
        if file is not None:
            keys.update((rel, locator) for locator in file.by_locator)
    named = set()
    for key in keys:
        if graph.node(key) is not None:
            named.update(target for _, target in graph.edges(key))
    roots = {key for key in keys - named
             if graph.node(key) is not None and graph.node(key).kind in ("fn", "macro")}
    rel, _, name = test.partition("::")
    file = build.file(rel)
    if file is not None:
        roots.update((rel, node.locator) for node in evidence_reach._walk(file.root)
                     if node.kind == "fn" and node.name == name)
    files = {key[0] for key in keys}
    return roots | evidence_reach.unseen_roots(graph, files), evidence_reach.unseen_macros(graph, files)


def _executed(build: Build, regions: dict[str, set]) -> dict:
    """Retained executed regions as `(file, locator) -> [character offsets]`."""
    found: dict = {}
    for path, points in regions.items():
        parts = path.strip("/").split("/")
        rel = next(("/".join(parts[i:]) for i in range(len(parts)) if (build.source / "/".join(parts[i:])).is_file()),
                   None)
        file = build.file(rel) if rel and rel.endswith(".rs") else None
        if file is None:
            continue
        for line, column in points:
            offset = file.offset(line, column)
            node = file.innermost(offset)
            if node is not file.root:
                found.setdefault((rel, node.locator), []).append(offset)
    return found


def crosscheck(build: Build, bound: Bound, found, regions: dict) -> int:
    """Compare the analysis with a retained run of the record: every function the run
    executed, among those the record binds whole, must be one the analysis enters, and
    each executed position must lie in a region it left open. Returns how many functions
    were compared; raises `Kept` on a contradiction."""
    graph, compared = build.graph, 0
    for key, offsets in _executed(build, regions).items():
        node = graph.node(key)
        if node is None or node.kind != "fn" or bound.allows(key[0], key[1], "full") != "full":
            continue
        compared += 1
        if key not in found.entered or any(
                graph.body(key).region_at(o) in found.closed.get(key, ()) for o in offsets):
            raise Kept("a retained run executed code the analysis closed")
    return compared


def select(build: Build, bound: Bound, found):
    """The narrowed selection of one record, computed in the build's source: the item
    parts, and the scope guards per file. `found` is the reachability of a sliced
    measurement; without it the selection is the record's own, with the files bound
    whole reduced to the items it uses."""
    selection = Selection(build.source, build.inventory, callee_bodies=True, within=bound.allows)
    selection.reach = found
    for rel in bound.whole:
        if rel not in bound.compiled:
            selection.keep_whole(rel, bound.reasons.get(rel, "bound whole"))
    for rel, parts in bound.parts.items():
        file = selection.file(rel)
        if file is None:
            continue
        for locator, have in parts.items():
            node = file.by_locator.get(locator)
            if node is None:
                continue
            if found is None:
                for part in have:
                    if part != "inner":
                        selection.need(rel, node, part)
            # A sliced measurement keeps the functions it can enter and the macro
            # invocations code that runs names; every other item is reached through them.
            elif "full" in have and ((node.kind == "fn" and (rel, locator) in found.entered)
                                     or (node.kind == "macro" and (rel, locator) in found.named)):
                selection.need(rel, node, "full")
    if found is not None:
        for rel in bound.compiled:
            file = selection.file(rel)
            for locator, node in (file.by_locator.items() if file is not None else ()):
                if node.kind == "fn" and (rel, locator) in found.entered:
                    selection.need(rel, node, "full")
    if selection.observes:
        raise Kept("the narrowed selection observes source positions")
    scopes: dict[str, list[list]] = {}
    for module_key, traits in sorted(selection._scope_closure().items()):
        module = build.inventory.modules.get(module_key)
        if module is None:
            continue
        scopes.setdefault(module["file"], []).append([module["chain"], traits])
        if not module["chain"]:
            for rel, owner in sorted(selection._includes.items()):
                if owner == module_key and [[], traits] not in scopes.setdefault(rel, []):
                    scopes[rel].append([[], traits])
    return selection, scopes


def _select_of(table: list[str], parts: dict[str, set[str]]) -> dict:
    position = {locator: i for i, locator in enumerate(table)}
    chosen = {}
    for part in PARTS:
        indices = sorted(position[locator] for locator, have in parts.items()
                         if locator in position and part in have
                         and (part in ("full", "inner") or "full" not in have))
        if part == "inner" and "inner" in parts.get("", ()):
            indices = [-1] + indices
        if indices:
            chosen[part] = indices
    return chosen


_DROP = object()
_ORDINAL = re.compile(r"#\d+(?=/|\Z)")


def same_names(source, table: list[str], selected) -> bool:
    """Whether the selected locators name the same items in a build's file (`source`, its
    locators) as in the table a manifest binds: each is present, and it and each item
    around it has as many same-named siblings in both. An ordinal can then not have moved
    from one item to another."""
    def census(locators) -> collections.Counter:
        return collections.Counter(_ORDINAL.sub("", locator) for locator in locators)
    here, there = census(source), census(table)
    for locator in selected:
        if not locator:
            continue
        if locator not in source:
            return False
        parts = locator.split("/")
        for depth in range(1, len(parts) + 1):
            name = _ORDINAL.sub("", "/".join(parts[:depth]))
            if here[name] != there[name]:
                return False
    return True


def narrowed_inputs(root: Path, checker: Checker, build: Build, bound: Bound, manifest: dict,
                    selection: Selection, scopes: dict) -> tuple[dict, dict[str, bytes], list[str], list[str]]:
    """The manifest's Rust inputs, narrowed: `(kind, path) -> new entry` (or `_DROP`), the
    item tables the new entries name, the files still bound whole, and the files in which
    a kept item reads differently in the build's source (the record was rebound to it after
    it was earned). Raises `Kept` when a check fails."""
    old_names = Names.load(bound.names) if bound.names else None
    # The guards are the narrowed selection's own (the names its items look up, the scopes
    # they resolve in, the method resolutions another declaration could change), as earning
    # writes them. A file that was bound whole had none: its items now need theirs.
    names = Names.of(selection.names)
    tables: dict[str, bytes] = {}
    changes: dict = {}
    whole: list[str] = []
    rebound: list[str] = []

    class inputs:  # the entries this pass replaces, by the entry they replace
        @staticmethod
        def append(new: dict) -> None:
            changes[(entry["kind"], entry["path"])] = new

    for entry in manifest["inputs"]:
        kind, rel = entry["kind"], entry.get("path")
        if kind not in ("rust_items", "evidence_file", "rust_names", "rust_guards"):
            continue
        changes[(kind, rel)] = _DROP
        if kind == "rust_items":
            source, current = build.file(rel), _parse(root / rel)
            # The analysis reads the build's source, the manifest names items by locator: the
            # two must name the same items. Either the source holds every bound part and
            # scope exactly as bound, or (a record rebound after an edit it cannot depend
            # on) every bound item is in the source under the same name, with as many
            # same-named siblings; an item whose text changed is then never kept (checked
            # below).
            try:
                as_built = source is not None and (
                    items_digest(source, bound.tables[rel], entry, old_names) == entry["digest"]
                    or same_names(source.by_locator, bound.tables[rel], bound.parts[rel]))
            except (KeyError, IndexError):
                as_built = False
            if not as_built:
                raise Kept("the build's source does not hold the bound items under the bound names")
            if rel in selection.whole:
                inputs.append(entry)
                continue
            have = bound.parts[rel]
            parts = {locator: {part for part in chosen if part in have.get(locator, ()) or (
                "full" in have.get(locator, ()) and part != "inner")}
                for locator, chosen in selection.parts.get(rel, {}).items()}
            parts = {locator: chosen for locator, chosen in parts.items() if chosen}
            new = {"kind": "rust_items", "path": rel, "role": entry["role"], "file": entry["file"],
                   "table": entry["table"], "select": _select_of(bound.tables[rel], parts),
                   "scopes": scopes.get(rel, [])}
            if not new["select"] and not new["scopes"]:
                continue
            try:
                new["digest"] = items_digest(current, bound.tables[rel], new, names)
                if items_digest(source, bound.tables[rel], new, names) != new["digest"]:
                    # Kept items the record was rebound to after it was earned: bound as the
                    # old manifest binds them, which is what the record is current against.
                    rebound.append(rel)
            except (KeyError, IndexError, AttributeError) as error:
                raise Kept(f"a selected item is gone from {rel}: {error}") from error
            inputs.append(new)
        elif kind == "evidence_file" and rel in bound.compiled:
            source, current = build.file(rel), _parse(root / rel)
            parts = selection.parts.get(rel, {})
            new = None
            if rel not in selection.whole and source is not None and current is not None:
                table = list(current.by_locator)
                new = {"kind": "rust_items", "path": rel, "role": "compile",
                       "file": checker.input_digest("file", rel), "table": sha(evidence_items.canonical(table)),
                       "select": _select_of(table, parts), "scopes": scopes.get(rel, [])}
                # The record is current, so the checkout's file is the one it binds whole: any
                # item of it is bound. The items the measurement uses are named in the build's
                # source, and each name must mean the same item in the checkout. One that
                # reads differently there was rebound after earning, as for any other file.
                named = same_names(source.by_locator, table, parts) and all(
                    current.module(chain) is not None and source.module(chain) is not None
                    for chain, _ in new["scopes"])
                if not named:
                    new = None
                elif any(current.digest(locator, part) != source.digest(locator, part)
                         for locator, chosen in parts.items() for part in chosen) or any(
                        scope_entries(current, current.module(chain), names, traits)
                        != scope_entries(source, source.module(chain), names, traits)
                        for chain, traits in new["scopes"]):
                    rebound.append(rel)
            if new is None:
                inputs.append(entry)
                whole.append(f"{rel}: {selection.whole.get(rel, bound.reasons.get(rel, 'bound whole'))}")
                continue
            if not new["select"] and not new["scopes"]:
                continue  # nothing the measurement uses is in this file
            tables[new["table"]] = evidence_items.canonical(table)
            new["digest"] = items_digest(current, table, new, names)
            inputs.append(new)
        elif kind == "evidence_file" and rel.endswith(".rs") and rel in bound.whole:
            inputs.append(entry)
            if rel in bound.reasons:
                whole.append(f"{rel}: {bound.reasons[rel]}")
        elif kind == "rust_names":
            inputs.append({**entry, **names.entry(), "digest": sha(canonical(names.entry()))})
        elif kind == "rust_guards":
            spec = selection.guard_spec(entry["dirs"])
            results = evidence_items.guard_results(root, checker, spec, entry["dirs"])
            inputs.append({**entry, "spec": spec, "tree": evidence_items.guard_tree(checker, entry["dirs"]),
                           "digest": sha(canonical(results))})
        else:
            inputs.append(entry)
    # A scope the selection resolves names in, in a file none of whose items it keeps.
    for rel, wanted in sorted(scopes.items()):
        if ("rust_items", rel) in changes or rel in bound.whole or rel in selection.whole:
            continue
        current = _parse(root / rel)
        if current is None:
            raise Kept(f"a scope of the selection is in a file this parser cannot read: {rel}")
        table = list(current.by_locator)
        new = {"kind": "rust_items", "path": rel, "role": "compile", "file": checker.input_digest("file", rel),
               "table": sha(evidence_items.canonical(table)), "select": {}, "scopes": wanted}
        try:
            new["digest"] = items_digest(current, table, new, names)
        except KeyError as error:
            raise Kept(f"a scope of the selection is gone from {rel}: {error}") from error
        tables[new["table"]] = evidence_items.canonical(table)
        changes[("rust_items", rel)] = new
    return changes, tables, whole, rebound


def _parse(path: Path):
    try:
        return parse(path)
    except (ParseError, OSError):
        return None


def recipes_by_content(root: Path, checker: Checker, manifest: dict, inputs: list[dict]) -> tuple[list[dict], int]:
    """Rebind the build's recipes and policy code by content. Each must hash, in the
    checkout, to the bytes the manifest bound."""
    from evidence_platforms import receipt_platform
    try:
        platform = receipt_platform(manifest["nix_build"])
        release = manifest["nix_build"].get("release", {}).get("recipe_sha256", {})
    except (KeyError, TypeError, ValueError):
        return inputs, 0
    recipes = {*platform.recipes, *release, *platform.policy_sources, "scripts/compile_time_environment.py"}
    texts = [(root / rel).read_text(encoding="utf-8", errors="replace")
             for rel in sorted(recipes) if rel.endswith(".nix") and (root / rel).is_file()]
    out, moved = [], 0
    for entry in inputs:
        rel = entry.get("path")
        if entry["kind"] == "evidence_file" and rel in recipes and (root / rel).is_file() \
                and checker.input_digest("file", rel) == entry["digest"]:
            recipe = evidence_recipes.entry(rel, (root / rel).read_bytes(), entry["role"], recipes=texts)
            if recipe is not None and recipe["raw"] == entry["digest"]:
                out.append(recipe)
                moved += 1
                continue
        out.append(entry)
    return out, moved


def _sliced(kind: str, manifest: dict) -> bool:
    """Whether the record's executed regions came from a slice of its runs, so that it
    binds the whole of every function its code names (a calibration measurement whose
    replicates ran on the uninstrumented build)."""
    return kind != "scientific" and "release" in (manifest.get("nix_build") or {})


def _plan(root: Path, checker: Checker, build: Build, bound: Bound, manifest: dict, test: str, sliced: bool):
    """The narrowing of one Rust selection: shared by the records that bind the same
    items through the same test."""
    found = None
    if sliced:
        roots, unseen = _roots(build, bound, test)
        found = evidence_reach.reach(build.graph, roots, unseen=unseen)
    selection, scopes = select(build, bound, found)
    changes, tables, whole, rebound = narrowed_inputs(root, checker, build, bound, manifest, selection, scopes)
    return found, changes, tables, whole, rebound


def narrow(root: Path, checker: Checker, binding: dict, ref: str, test: str, *, kind: str,
           regions: dict | None = None, write: Path | None = None) -> dict:
    """Narrow one current record: its new manifest address and counts. `kept` says why the
    Rust selection stayed as it was (if it did). The manifest is stored under `write` (the
    repository, or a staging directory laid out like it) when one is given."""
    manifest = checker.load(ref)
    bound = Bound(checker, manifest)
    result = {"old": ref, "kept": None, "compared": None, "rebound_files": 0}
    inputs, tables, whole = manifest["inputs"], {}, manifest.get("whole_sources", [])
    build = None
    try:
        build = build_of(manifest)
        sliced = _sliced(kind, manifest)
        key = digest(canonical([test, sliced, [entry for entry in manifest["inputs"]
                                               if entry["kind"] in ("rust_items", "rust_names", "rust_guards")
                                               or str(entry.get("path", "")).endswith(".rs")]]))
        if key not in build.memo:
            if len(build.memo) >= 16:
                build.memo.pop(next(iter(build.memo)))
            try:
                build.memo[key] = _plan(root, checker, build, bound, manifest, test, sliced)
            except Kept as kept:
                build.memo[key] = kept
        plan = build.memo[key]
        if isinstance(plan, Kept):
            raise plan
        found, changes, tables, whole, rebound = plan
        result["rebound_files"] = len(rebound)
        if regions is not None and found is not None:
            result["compared"] = crosscheck(build, bound, found, regions)
        present = {(entry["kind"], entry.get("path")) for entry in manifest["inputs"]}
        inputs = [changes.get((entry["kind"], entry.get("path")), entry) for entry in manifest["inputs"]]
        inputs += [entry for key, entry in changes.items() if key not in present]
        inputs = [entry for entry in inputs if entry is not _DROP]
    except Kept as kept:
        result["kept"] = str(kept)
        inputs, tables, whole = manifest["inputs"], {}, manifest.get("whole_sources", [])
    result["before"] = bound.count(build)
    inputs, result["recipes"] = recipes_by_content(root, checker, manifest, inputs)
    if inputs == manifest["inputs"]:
        result.update(new=ref, after=result["before"])
        return result
    new = {name: value for name, value in manifest.items() if name != "inputs"}
    # The manifest the record was earned with: kept through a second narrowing.
    new.update(inputs=inputs, analyzer=ANALYZER, whole_sources=whole,
               narrowed_from=manifest.get("narrowed_from", ref))
    manifests.validate(new, binding)
    stored, groups = grouped(new)
    new_ref = manifest_id(stored)
    # Check the new manifest as any other, before anything is written.
    checker.manifests[new_ref] = new
    checker.items.pending = {**checker.items.pending, **tables}
    status = checker.check(binding, new_ref)
    if not status["current"]:
        raise RuntimeError(f"narrowed manifest of {binding['key']} is not current: {status['reasons'][:3]}")
    result.update(new=new_ref, after=Bound(checker, new).count(build))
    del checker.manifests[new_ref]
    if write is not None:
        from evidence_earning import _write_addressed
        for address, data in tables.items():
            _write_addressed(write / evidence_items.STORE, address, data, write)
        for address, data in groups.items():
            _write_addressed(write / manifests.INPUT_GROUPS, address, data, write)
        _write_addressed(write / manifests.STORE, new_ref, canonical(stored), write)
    return result


# ---------------------------------------------------------------------- the registries

def records(root: Path, kind: str) -> list[tuple[str, dict, str, str]]:
    """(kind, binding, manifest address, test) of every record that cites a manifest."""
    from evidence_earning import binding, measurement_binding, probe_binding
    found = []
    for category, (filename, table) in REGISTRIES.items():
        if kind not in ("all", category) or not (root / filename).is_file():
            continue
        for row in _repo.load_toml(root / filename).get(table, []):
            ref = row.get("dependency_manifest")
            if not isinstance(ref, str):
                continue
            if category == "scientific":
                bound = binding("scientific", row.get("test"), row.get("fingerprint"))
            elif category == "calibration":
                bound = measurement_binding(row)
            else:
                bound = probe_binding(row)
            found.append((category, bound, ref, str(row.get("test", ""))))
    return found


def retained_regions(root: Path, log_dirs: list[Path]) -> dict[str, tuple[str, dict]]:
    """Record id -> (build output, executed regions) from retained measurement logs."""
    from calibration_runner import BUILDS, record_runs
    found: dict[str, tuple[str, dict]] = {}
    for log_dir in log_dirs:
        if not (log_dir / BUILDS).is_file():
            continue
        builds = json.loads((log_dir / BUILDS).read_text(encoding="utf-8"))
        outputs = {build["artifact"]["output"] for build in builds.values()}
        if len(outputs) != 1:
            continue
        for rid, stems in record_runs(log_dir).items():
            regions: dict[str, set] = {}
            try:
                for stem in stems:
                    for source, points in json.loads((log_dir / f"{stem}.coverage.json").read_text(encoding="utf-8")).items():
                        regions.setdefault(source, set()).update(tuple(p) for p in points)
            except (OSError, ValueError):
                continue
            found[rid] = (next(iter(outputs)), regions)
    return found


_WORK: dict = {}


def _one(index: int) -> dict:
    root, checker, items, retained = _WORK["root"], _WORK["checker"], _WORK["items"], _WORK["retained"]
    kind, binding, ref, test = items[index]
    regions = None
    manifest = checker.load(ref)
    held = retained.get(binding["key"])
    if held is not None and held[0] == (manifest.get("nix_build") or {}).get("artifact", {}).get("output"):
        regions = held[1]
    try:
        result = narrow(root, checker, binding, ref, test, kind=kind, regions=regions, write=_WORK["write"])
    except Exception as error:  # one record's failure leaves that record as it is
        result = {"old": ref, "new": ref, "kept": f"not narrowed: {type(error).__name__}: {error}"[:200],
                  "compared": None, "rebound_files": 0, "before": 0, "after": 0, "recipes": 0}
    result.update(kind=kind, key=binding["key"])
    if index % 250 == 0:
        print(f"record {index} of {len(items)}", file=sys.stderr, flush=True)
    return result


STAGED = "narrowing.json"


def run(root: Path, kind: str, *, apply: bool, log_dirs: list[Path], jobs: int, only: str | None = None,
        stage: Path | None = None, out=sys.stdout) -> dict:
    """Narrow every current record. A dry run writes nothing, or, with `stage`, writes the
    new manifests and the address each record would cite under that directory, from which
    `apply_staged` rebinds the records without computing them again."""
    checker = Checker(root)
    found = [item for item in records(root, kind) if only is None or only in item[1]["key"]]
    current, refused = [], []
    for item in found:
        status = checker.check(item[1], item[2])
        (current if status["current"] else refused).append(item)
    # Records of one build together, so a worker reads each inventory once.
    current.sort(key=lambda item: (str((checker.load(item[2]).get("nix_build") or {}).get("artifact", {}).get("output")),
                                   item[3], item[1]["key"]))
    if stage is not None:
        stage.mkdir(parents=True, exist_ok=True)
    _WORK.update(root=root, checker=checker, items=current, retained=retained_regions(root, log_dirs),
                 write=root if apply else stage)
    print(f"{len(current)} current record(s) to narrow, {len(refused)} refused", file=sys.stderr, flush=True)
    if jobs > 1 and len(current) > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            results = pool.map(_one, range(len(current)), chunksize=max(1, min(64, len(current) // (jobs * 4) or 1)))
    else:
        results = [_one(index) for index in range(len(current))]
    summary: dict = {"records": len(found), "refused_not_current": len(refused), "by_kind": {}}
    for category in sorted({r["kind"] for r in results}):
        rows = [r for r in results if r["kind"] == category]
        moved = [r for r in rows if r["new"] != r["old"]]
        kept = collections.Counter(r["kept"] for r in rows if r["kept"])
        compared = [r["compared"] for r in rows if r["compared"] is not None]
        summary["by_kind"][category] = {
            "current": len(rows), "migrated": len(moved),
            "selection_narrowed": sum(1 for r in rows if r["kept"] is None and r["after"] < r["before"]),
            "mean_items_before": round(sum(r["before"] for r in rows) / len(rows), 1),
            "mean_items_after": round(sum(r["after"] for r in rows) / len(rows), 1),
            "recipes_bound_by_content": sum(r["recipes"] for r in rows),
            "keeping_items_rebound_since_earning": sum(1 for r in rows if r["rebound_files"]),
            "left_broad": dict(kept),
            "cross_checked_records": len(compared), "cross_checked_functions": sum(compared),
        }
    moved = {result["old"]: result["new"] for result in results if result["new"] != result["old"]}
    if apply:
        _cite(root, moved)
        summary["pruned"] = len(manifests.prune_evidence(root))
    elif stage is not None:
        (stage / STAGED).write_text(json.dumps({"moved": moved, "summary": summary}, sort_keys=True), encoding="utf-8")
    json.dump(summary, out, indent=2, sort_keys=True)
    out.write("\n")
    for _, binding, _, _ in refused[:20]:
        out.write(f"refused (not current): {binding['kind']}:{binding['key']}\n")
    return summary


def apply_staged(root: Path, stage: Path, out=sys.stdout) -> dict:
    """Rebind the records to the manifests a dry run staged. A record whose address moved
    since, or is no longer current, is left alone; every staged manifest is copied and then
    checked against the checkout, as any manifest is, before its record cites it."""
    from evidence_earning import _write_addressed
    moved = json.loads((stage / STAGED).read_text(encoding="utf-8"))["moved"]
    checker = Checker(root)
    wanted = [(binding, ref) for _, binding, ref, _ in records(root, "all")
              if ref in moved and checker.check(binding, ref)["current"]]
    for store in (manifests.STORE, manifests.INPUT_GROUPS, evidence_items.STORE):
        for path in sorted((stage / store).glob("*.json")) if (stage / store).is_dir() else ():
            _write_addressed(root / store, path.stem, path.read_bytes(), root)
    fresh = Checker(root)
    failed = [binding["key"] for binding, ref in wanted if not fresh.check(binding, moved[ref])["current"]]
    if failed:
        manifests.prune_evidence(root)
        raise SystemExit(f"{len(failed)} staged manifest(s) are not current; nothing was rebound: {failed[:5]}")
    _cite(root, {ref: moved[ref] for _, ref in wanted})
    summary = {"rebound": len(wanted), "staged": len(moved), "pruned": len(manifests.prune_evidence(root))}
    json.dump(summary, out, indent=2, sort_keys=True)
    out.write("\n")
    return summary


def _cite(root: Path, moved: dict[str, str]) -> None:
    """Cite each new manifest from its record. Only the address changes."""
    pattern = re.compile(r'^(dependency_manifest = ")([0-9a-f]{64})(")$', re.M)
    for filename, _ in REGISTRIES.values():
        path = root / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        rewritten = pattern.sub(lambda m: m.group(1) + moved.get(m.group(2), m.group(2)) + m.group(3), text)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("all", *REGISTRIES), default="all")
    parser.add_argument("--apply", action="store_true", help="write the manifests and registries (default: dry run)")
    parser.add_argument("--logs", action="append", default=[], type=Path,
                        help="a directory of retained measurement logs to cross-check against (repeatable)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--only", help="only records whose key contains this text")
    parser.add_argument("--stage", type=Path,
                        help="a directory: a dry run keeps its manifests there; with --apply, rebind from it")
    args = parser.parse_args(argv)
    if args.apply and args.stage is not None:
        apply_staged(ROOT, args.stage)
    else:
        run(ROOT, args.kind, apply=args.apply, log_dirs=args.logs, jobs=args.jobs, only=args.only, stage=args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
