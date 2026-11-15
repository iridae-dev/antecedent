"""Rebinding an earned record to the narrower selection, without re-measuring.

A small workspace with a dispatcher and two routes stands for the registry: a record
earned by one route is bound, as a sliced measurement was, to both routes' functions.
"""
from __future__ import annotations

import io
import json
import shutil
import unittest
from unittest import mock

import evidence_items as ei
import evidence_manifests as em
import narrow_evidence_binding as narrowing
from _repo import load_toml
from test_evidence_reach import LIB, ROUTES, SOURCES, TEST, Workspace, inventory

TEST_KEY = "c::t::measures_alpha"
REGISTRY = "parity/scientific_records.toml"
PIN = {"release": "1.0.0", "commit": "0" * 40}


class NarrowingTests(Workspace):
    def setUp(self):
        super().setUp()
        self.write("tools/evidence-analyzer/pin.toml", 'release = "1.0.0"\ncommit = "%s"\n' % ("0" * 40))
        # The build the record was earned from: its source tree and compiler inventory.
        self.build_dir = self.root / "build"
        shutil.copytree(self.root / "crates", self.build_dir / "source" / "crates")
        (self.build_dir / "output" / "compiler-inventory").mkdir(parents=True)
        self.old = self.earn_record()
        self.old_parts = self.selected(self.old)
        self.build = narrowing.Build(self.build_dir / "output", self.build_dir / "source")
        patches = (mock.patch.object(narrowing, "build_of", lambda manifest: self.build),
                   mock.patch.object(narrowing, "_sliced", lambda kind, manifest: True))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def earn_record(self) -> str:
        """A record bound as a sliced measurement was: every named function whole."""
        (self.build_dir / "output" / "compiler-inventory" / "items.jsonl").write_text(
            "\n".join(json.dumps(record) for record in _inventory_records()), encoding="utf-8")
        _, entries = self.earn(narrow=False)
        binding = dict(kind="scientific", key=TEST_KEY, fingerprint="f" * 64, measurement=None)
        manifest = dict(schema=em.SCHEMA, normalization=em.NORMALIZATION, record=binding, compiler=PIN,
                        analyzer="compiler-items-v1", inputs=entries, fallbacks=[], whole_sources=[],
                        declarations={})
        stored, groups = em.grouped(manifest)
        for address, data in groups.items():
            self.write(f"{em.INPUT_GROUPS}/{address}.json", data.decode())
        ref = em.manifest_id(stored)
        self.write(f"{em.STORE}/{ref}.json", em.canonical(stored).decode())
        self.write(REGISTRY, f'[[record]]\ntest = "{TEST_KEY}"\nfingerprint = "{"f" * 64}"\npassed = true\n'
                             f'observed = 0.951\ndependency_manifest = "{ref}"\n')
        return ref

    def run_tool(self, *, apply: bool) -> dict:
        em._FILE_DIGESTS.clear()
        return narrowing.run(self.root, "scientific", apply=apply, log_dirs=[], jobs=1, out=io.StringIO())

    def cited(self) -> str:
        return load_toml(self.root / REGISTRY)["record"][0]["dependency_manifest"]

    def status(self) -> dict:
        em._FILE_DIGESTS.clear()
        return em.Checker(self.root, {}).report(em.inventory(self.root, "scientific"))["records"][0]

    def selected(self, ref: str) -> dict[str, dict[str, set]]:
        checker = em.Checker(self.root, {})
        return narrowing.Bound(checker, checker.load(ref)).parts

    def test_a_dry_run_counts_and_writes_nothing(self):
        before = (self.root / REGISTRY).read_text(encoding="utf-8")
        summary = self.run_tool(apply=False)["by_kind"]["scientific"]
        self.assertEqual((summary["current"], summary["migrated"], summary["selection_narrowed"]), (1, 1, 1))
        self.assertLess(summary["mean_items_after"], summary["mean_items_before"])
        self.assertEqual(summary["left_broad"], {})
        self.assertEqual((self.root / REGISTRY).read_text(encoding="utf-8"), before)
        self.assertEqual([p.stem for p in (self.root / em.STORE).iterdir()], [self.old])

    def test_the_narrowed_manifest_is_a_subset_and_only_the_address_changes(self):
        before = (self.root / REGISTRY).read_text(encoding="utf-8")
        self.run_tool(apply=True)
        new = self.cited()
        self.assertNotEqual(new, self.old)
        self.assertEqual((self.root / REGISTRY).read_text(encoding="utf-8"), before.replace(self.old, new))
        self.assertTrue(self.status()["current"])
        new_parts = self.selected(new)
        for rel, parts in new_parts.items():
            for locator, chosen in parts.items():
                have = self.old_parts[rel][locator]
                self.assertTrue(chosen <= have or "full" in have, (rel, locator))
        self.assertNotIn("fn:invert#0", new_parts[ROUTES])
        self.assertNotIn("fn:beta#0", new_parts[ROUTES])
        self.assertIn("full", new_parts[ROUTES]["fn:alpha#0"])
        manifest = json.loads((self.root / em.STORE / f"{new}.json").read_bytes())
        self.assertEqual(manifest["record"]["fingerprint"], "f" * 64)
        self.assertEqual(manifest["narrowed_from"], self.old)
        self.assertEqual(manifest["analyzer"], em.ANALYZER)
        # The superseded manifest is no longer reachable and is removed.
        self.assertFalse((self.root / em.STORE / f"{self.old}.json").exists())

    def test_a_staged_dry_run_is_applied_without_computing_it_again(self):
        stage = self.root / "stage"
        before = (self.root / REGISTRY).read_text(encoding="utf-8")
        em._FILE_DIGESTS.clear()
        narrowing.run(self.root, "scientific", apply=False, log_dirs=[], jobs=1, stage=stage, out=io.StringIO())
        self.assertEqual((self.root / REGISTRY).read_text(encoding="utf-8"), before)
        summary = narrowing.apply_staged(self.root, stage, out=io.StringIO())
        self.assertEqual((summary["rebound"], summary["staged"]), (1, 1))
        self.assertNotEqual(self.cited(), self.old)
        self.assertTrue(self.status()["current"])
        self.assertNotIn("fn:invert#0", self.selected(self.cited())[ROUTES])

    def test_a_staged_manifest_that_is_no_longer_current_rebinds_nothing(self):
        stage = self.root / "stage"
        em._FILE_DIGESTS.clear()
        narrowing.run(self.root, "scientific", apply=False, log_dirs=[], jobs=1, stage=stage, out=io.StringIO())
        self.edit(ROUTES, "q.x + shared(1.0)", "q.x - shared(1.0)")
        summary = narrowing.apply_staged(self.root, stage, out=io.StringIO())
        self.assertEqual(summary["rebound"], 0)
        self.assertEqual(self.cited(), self.old)

    def test_rebinding_recipes_only_leaves_the_rust_selection_as_it_is(self):
        before = (self.root / REGISTRY).read_text()
        summary = narrowing.run(self.root, "scientific", apply=True, log_dirs=[], jobs=1, out=io.StringIO(),
                                recipes_only=True)["by_kind"]["scientific"]
        self.assertEqual((summary["current"], summary["migrated"], summary["selection_narrowed"]), (1, 0, 0))
        self.assertEqual(summary["left_broad"], {})
        self.assertEqual((self.root / REGISTRY).read_text(), before)

    def test_a_recipe_bound_by_its_bytes_is_rebound_by_what_it_says(self):
        from types import SimpleNamespace
        recipe = "nix/evidence-target.nix"
        self.write(recipe, "{ a = 1; } # a comment\n")
        checker = em.Checker(self.root, {})
        raw = checker.input_digest("file", recipe)
        manifest = {"nix_build": {}}
        platform = SimpleNamespace(recipes=(recipe,), policy_sources=())
        entry = {"kind": "evidence_file", "path": recipe, "role": "external", "digest": raw}
        with mock.patch("evidence_platforms.receipt_platform", lambda receipt: platform):
            inputs, moved = narrowing.recipes_by_content(self.root, checker, manifest, [entry])
            self.assertEqual((moved, inputs[0]["kind"], inputs[0]["raw"]), (1, "build_recipe", raw))
            # Bytes that no longer match what the manifest bound are not rebound.
            stale = dict(entry, digest="0" * 64)
            self.assertEqual(narrowing.recipes_by_content(self.root, checker, manifest, [stale]), ([stale], 0))
        # A comment edited afterwards owes nothing: the content digest is the same.
        self.write(recipe, "# edited\n{ a = 1; }\n")
        em._FILE_DIGESTS.clear()
        self.assertTrue(em.Checker(self.root, {}).recipe_current(inputs[0]))

    def test_a_narrowed_record_is_left_as_it_is(self):
        self.run_tool(apply=True)
        cited = self.cited()
        summary = self.run_tool(apply=True)["by_kind"]["scientific"]
        self.assertEqual((summary["current"], summary["migrated"]), (1, 0))
        self.assertEqual(self.cited(), cited)

    def test_only_the_measured_route_owes_the_record_afterwards(self):
        self.run_tool(apply=True)
        self.edit(ROUTES, "{ 1.0 / y }", "{ 2.0 / y }")       # the route the record never enters
        self.assertTrue(self.status()["current"])
        self.edit(ROUTES, "q.x + shared(1.0)", "q.x - shared(1.0)")
        status = self.status()
        self.assertFalse(status["current"])
        self.assertEqual(status["reasons"], [f"input_changed:rust_items:{ROUTES}"])

    def test_a_record_that_is_not_current_is_refused(self):
        self.edit(ROUTES, "q.x + shared(1.0)", "q.x - shared(1.0)")
        before = (self.root / REGISTRY).read_text(encoding="utf-8")
        summary = self.run_tool(apply=True)
        self.assertEqual(summary["refused_not_current"], 1)
        self.assertEqual(summary["by_kind"], {})
        self.assertEqual((self.root / REGISTRY).read_text(encoding="utf-8"), before)
        self.assertEqual(self.cited(), self.old)

    def test_a_kept_item_rebound_after_earning_stays_bound_as_the_manifest_binds_it(self):
        # The record was rebound to an edit of a function it keeps (the build's source
        # holds the earlier text). The narrowed manifest binds the function as the old
        # one does, and the tool says so.
        path = self.build_dir / "source" / ROUTES
        path.write_text(path.read_text(encoding="utf-8").replace("q.x + shared(1.0)", "q.x * shared(1.0)"), encoding="utf-8")
        ei._PARSED.clear()
        summary = self.run_tool(apply=True)["by_kind"]["scientific"]
        self.assertEqual(summary["keeping_items_rebound_since_earning"], 1)
        self.assertTrue(self.status()["current"])
        self.edit(ROUTES, "q.x + shared(1.0)", "q.x * shared(1.0)")
        self.assertFalse(self.status()["current"])

    def test_a_bound_name_that_means_another_item_in_the_built_source_keeps_the_record_broad(self):
        # The build's source has a second `alpha` before the bound one: `fn:alpha#0` is a
        # different function there, so the analysis cannot be read against the manifest.
        path = self.build_dir / "source" / ROUTES
        path.write_text(path.read_text(encoding="utf-8").replace("pub fn alpha", "#[cfg(any())]\nfn alpha() {}\n\npub fn alpha"),
                        encoding="utf-8")
        ei._PARSED.clear()
        summary = self.run_tool(apply=True)["by_kind"]["scientific"]
        self.assertEqual(summary["selection_narrowed"], 0)
        self.assertEqual(summary["left_broad"],
                         {"the build's source does not hold the bound items under the bound names": 1})
        self.assertEqual(self.cited(), self.old)

    def test_a_retained_run_that_contradicts_the_analysis_keeps_the_record_broad(self):
        checker = em.Checker(self.root, {})
        binding = dict(kind="scientific", key=TEST_KEY, fingerprint="f" * 64, measurement=None)
        line = SOURCES[ROUTES][:SOURCES[ROUTES].index("invert(q.y)")].count("\n") + 1
        result = narrowing.narrow(self.root, checker, binding, self.old, f"{TEST}::measures_alpha", kind="scientific",
                                  regions={f"/sandbox/{ROUTES}": {(line, 5)}})
        self.assertEqual(result["kept"], "a retained run executed code the analysis closed")
        self.assertEqual(result["new"], self.old)
        agreeing = narrowing.narrow(self.root, checker, binding, self.old, f"{TEST}::measures_alpha",
                                    kind="scientific", regions={f"/sandbox/{LIB}": {(11, 5)}})
        self.assertIsNone(agreeing["kept"])
        self.assertEqual(agreeing["compared"], 1)


def _inventory_records() -> list[dict]:
    """The records of `test_evidence_reach.inventory`, as the driver writes them."""
    found = inventory()
    records = [dict(module, t="mod") for module in found.modules.values()]
    for key, item in found.items.items():
        records.append({"t": "item", "key": key, "kind": item["kind"], "mod": item["mod"],
                        "deps": sorted(map(list, item["deps"])), "bdeps": sorted(map(list, item["bdeps"])),
                        "lookups": [], "probes": [], "observes": False, "includes": False})
    return records


class IncludedFileTests(unittest.TestCase):
    """A record that bound a file compiled in place whole is rebound to the items it uses."""

    LIB, HELPERS = "crates/c/src/lib.rs", "crates/c/src/helpers.rs"
    LIB_SOURCE = 'pub fn top() -> u8 { used() }\n\ninclude!("helpers.rs");\n'
    HELPERS_SOURCE = "fn used() -> u8 { 1 }\n\nfn idle() -> u8 { 2 }\n"

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        for rel, text in ((self.LIB, self.LIB_SOURCE), (self.HELPERS, self.HELPERS_SOURCE),
                          ("crates/c/Cargo.toml", "[package]\n"),
                          ("tools/evidence-analyzer/pin.toml", 'release = "1.0.0"\ncommit = "%s"\n' % ("0" * 40))):
            self.write(rel, text)
        ei._PARSED.clear()
        em._FILE_DIGESTS.clear()
        module = f"{self.LIB}#"
        records = [{"t": "mod", "key": module, "file": self.LIB, "chain": [], "parent": None, "name": ""}]
        for rel, source, needle, bdeps in (
                (self.LIB, self.LIB_SOURCE, "top", [["fn", f"{self.HELPERS}:{self.HELPERS_SOURCE.index('used')}"]]),
                (self.HELPERS, self.HELPERS_SOURCE, "used", []), (self.HELPERS, self.HELPERS_SOURCE, "idle", [])):
            records.append({"t": "item", "key": f"{rel}:{source.index(needle)}", "kind": "Fn", "mod": module,
                            "deps": [], "bdeps": bdeps, "lookups": [[module, "used"]], "probes": [],
                            "observes": False, "includes": False})
        build = self.root / "build"
        shutil.copytree(self.root / "crates", build / "source" / "crates")
        (build / "output" / "compiler-inventory").mkdir(parents=True)
        (build / "output" / "compiler-inventory" / "items.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8")
        # As the record was earned: the including file by item, the included file whole.
        checker = em.Checker(self.root, {})
        selection = ei.Selection(self.root, ei.Inventory.read(json.dumps(r) for r in records))
        selection.execute(self.LIB, self.LIB_SOURCE.index("used()"))
        entries, tables = selection.entries(checker, ["crates/c"], "crates/c")
        entries = [e for e in entries if e.get("path") != self.HELPERS]
        entries.append(dict(kind="evidence_file", path=self.HELPERS, role="compile", text="rust",
                            digest=checker.rust_text(self.HELPERS)))
        for address, data in tables.items():
            self.write(f"{ei.STORE}/{address}.json", data.decode())
        self.binding = dict(kind="scientific", key=TEST_KEY, fingerprint="f" * 64, measurement=None)
        manifest = dict(schema=em.SCHEMA, normalization=em.NORMALIZATION, record=self.binding, compiler=PIN,
                        analyzer="compiler-items-v1", inputs=entries, fallbacks=[], declarations={},
                        whole_sources=[f"{self.HELPERS}: no compiler module for this file"])
        stored, groups = em.grouped(manifest)
        for address, data in groups.items():
            self.write(f"{em.INPUT_GROUPS}/{address}.json", data.decode())
        self.old = em.manifest_id(stored)
        self.write(f"{em.STORE}/{self.old}.json", em.canonical(stored).decode())
        self.write(REGISTRY, f'[[record]]\ntest = "{TEST_KEY}"\nfingerprint = "{"f" * 64}"\n'
                             f'dependency_manifest = "{self.old}"\n')
        self.build = narrowing.Build(build / "output", build / "source")
        patch = mock.patch.object(narrowing, "build_of", lambda manifest: self.build)
        patch.start()
        self.addCleanup(patch.stop)

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def edit(self, rel, old, new):
        path = self.root / rel
        path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
        em._FILE_DIGESTS.clear()

    def rebind(self) -> dict:
        em._FILE_DIGESTS.clear()
        narrowing.run(self.root, "scientific", apply=True, log_dirs=[], jobs=1, out=io.StringIO())
        em._FILE_DIGESTS.clear()
        checker = em.Checker(self.root, {})
        ref = load_toml(self.root / REGISTRY)["record"][0]["dependency_manifest"]
        return next((e for e in checker.load(ref)["inputs"] if e.get("path") == self.HELPERS), None)

    def current(self) -> bool:
        em._FILE_DIGESTS.clear()
        return em.Checker(self.root, {}).report(em.inventory(self.root, "scientific"))["records"][0]["current"]

    def test_the_included_file_is_bound_by_the_items_the_record_uses(self):
        entry = self.rebind()
        self.assertEqual(entry["kind"], "rust_items")
        self.assertTrue(self.current())
        self.edit(self.HELPERS, "{ 2 }", "{ 3 }")          # a function the record does not use
        self.assertTrue(self.current())
        self.edit(self.HELPERS, "{ 1 }", "{ 4 }")          # the one it does
        self.assertFalse(self.current())

    def test_the_items_of_the_included_file_are_guarded_where_their_names_resolve(self):
        # Bound whole, the file had no guard on how its names resolve. By item it does: an
        # import of a name its kept function is written with, in the including module.
        self.rebind()
        self.edit(self.LIB, "include!", "use crate::elsewhere::used;\n\ninclude!")
        self.assertFalse(self.current())

    def test_a_file_that_changed_since_in_what_the_record_does_not_use_is_still_rebound(self):
        # The record was rebound, whole, to an edit of the unused function.
        self.edit(self.HELPERS, "{ 2 }", "{ 3 }")
        checker = em.Checker(self.root, {})
        manifest = checker.load(self.old)
        inputs = [dict(e, digest=checker.rust_text(self.HELPERS)) if e.get("path") == self.HELPERS else e
                  for e in manifest["inputs"]]
        stored, groups = em.grouped({**{k: v for k, v in manifest.items() if k != "inputs"}, "inputs": inputs})
        for address, data in groups.items():
            self.write(f"{em.INPUT_GROUPS}/{address}.json", data.decode())
        ref = em.manifest_id(stored)
        self.write(f"{em.STORE}/{ref}.json", em.canonical(stored).decode())
        self.write(REGISTRY, (self.root / REGISTRY).read_text(encoding="utf-8").replace(self.old, ref))
        self.assertEqual(self.rebind()["kind"], "rust_items")
        self.assertTrue(self.current())

    def test_a_name_that_means_another_item_in_the_built_source_keeps_the_file_whole(self):
        path = self.build.source / self.HELPERS
        path.write_text("#[cfg(any())]\nfn used() {}\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
        ei._PARSED.clear()
        self.assertEqual(self.rebind()["kind"], "evidence_file")
        self.assertTrue(self.current())


class SameNamesTests(unittest.TestCase):
    def test_an_added_item_does_not_move_a_bound_name(self):
        source = {"fn:a#0": 0, "impl:S#0": 0, "impl:S#0/fn:f#0": 0}
        self.assertTrue(narrowing.same_names(source, [*source, "const:NEW#0"], ["impl:S#0/fn:f#0", "fn:a#0"]))

    def test_a_same_named_sibling_or_a_missing_item_is_refused(self):
        source = {"fn:a#0": 0, "impl:S#0": 0, "impl:S#0/fn:f#0": 0}
        self.assertFalse(narrowing.same_names(source, [*source, "impl:S#1", "impl:S#1/fn:g#0"], ["impl:S#0/fn:f#0"]))
        self.assertFalse(narrowing.same_names(source, [*source, "fn:b#0"], ["fn:b#0"]))


if __name__ == "__main__":
    unittest.main()
