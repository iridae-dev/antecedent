"""Item-level evidence: the Rust item parser, selection from an inventory, and currency.

The inventories here are written by hand in the compiler driver's format, so these
checks need no compiler.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evidence_items as ei
import evidence_manifests as em
import rust_items as ri

LIB = "crates/c/src/lib.rs"
EST = "crates/c/src/est.rs"
TEST = "crates/c/tests/t.rs"

LIB_SOURCE = """#![allow(dead_code)]
pub mod est;
pub use est::*;

pub const SCALE: f64 = 2.0;

pub fn helper(x: f64) -> f64 { x + 1.0 }

pub fn unused() -> u32 { line!() }
"""
EST_SOURCE = """pub struct Mean { pub bias: f64 }

impl Mean {
    pub fn fit(&self) -> f64 { self.bias + crate::helper(crate::SCALE) }
    fn idle(&self) -> f64 { 0.0 }
}
"""
TEST_SOURCE = """use c::Mean;

#[test]
fn runs() {
    assert!(Mean { bias: 1.0 }.fit() > 0.0);
}
"""


def key(rel: str, source: str, needle: str) -> str:
    """A driver key: the byte offset of an item's name."""
    return f"{rel}:{len(source[:source.index(needle)].encode())}"


class ParserTests(unittest.TestCase):
    def test_locators_name_items_by_kind_name_and_container(self):
        file = ri.File(EST_SOURCE)
        self.assertEqual(list(file.by_locator),
                         ["struct:Mean#0", "impl:Mean#0", "impl:Mean#0/fn:fit#0", "impl:Mean#0/fn:idle#0"])

    def test_hashes_ignore_layout_and_comments_but_not_literals(self):
        base = ri.File('fn f() -> &str { "a b" }\n').digest("fn:f#0", "full")
        self.assertEqual(base, ri.File('// note\nfn  f()->&str{\n  "a b" }\n').digest("fn:f#0", "full"))
        self.assertNotEqual(base, ri.File('fn f() -> &str { "a  b" }\n').digest("fn:f#0", "full"))

    def test_a_signature_excludes_the_body(self):
        before = ri.File("fn f(x: u8) -> u8 { x }\n")
        self.assertEqual(before.digest("fn:f#0", "sig"), ri.File("fn f(x: u8) -> u8 { x + 1 }\n").digest("fn:f#0", "sig"))
        self.assertNotEqual(before.digest("fn:f#0", "sig"), ri.File("fn f(x: u16) -> u16 { x }\n").digest("fn:f#0", "sig"))

    def test_docs_count_only_under_a_macro_that_may_read_them(self):
        plain = "/// one\n#[derive(Debug)]\npub struct S;\n"
        self.assertEqual(ri.File(plain).digest("struct:S#0", "full"),
                         ri.File(plain.replace("one", "two")).digest("struct:S#0", "full"))
        custom = "/// one\n#[derive(Parser)]\npub struct S;\n"
        self.assertNotEqual(ri.File(custom).digest("struct:S#0", "full"),
                            ri.File(custom.replace("one", "two")).digest("struct:S#0", "full"))

    def test_use_bindings_and_impl_heads(self):
        self.assertEqual(ri.use_bindings("use a::{b as c, d::{self, e}, f::*};"), ({"c", "d", "e"}, True))
        self.assertEqual(ri.impl_parts("<T: X> Tr<T> for Vec<T> where T: Y"), ("Tr", "Vec", {"T"}))
        self.assertEqual(ri.impl_parts("<T> Tr for T")[1], "*")
        self.assertEqual(ri.impl_parts("Mean")[:2], (None, "Mean"))

    def test_generic_braces_and_where_clauses_do_not_end_a_header(self):
        file = ri.File("impl<F> T for W<F>\nwhere\n    F: Fn() -> u8,\n{\n    fn g(&self) {}\n}\n"
                       "struct S<T = ()> { t: T }\n")
        self.assertIn("impl:<F>T for W<F>where F:Fn()->u8,#0/fn:g#0", file.by_locator)
        self.assertIn("struct:S#0", file.by_locator)

    def test_an_unreadable_file_raises(self):
        with self.assertRaises(ri.ParseError):
            ri.File("fn f() {\n")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        for rel, text in ((LIB, LIB_SOURCE), (EST, EST_SOURCE), (TEST, TEST_SOURCE),
                          ("crates/c/Cargo.toml", '[package]\nname = "c"\n')):
            self.write(rel, text)
        ei._PARSED.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        em._FILE_DIGESTS.clear()

    def inventory(self, observes=False):
        lib, est, test = "crates/c/src/lib.rs#", "crates/c/src/lib.rs#::est", "crates/c/tests/t.rs#"
        mean, fit, helper = key(EST, EST_SOURCE, "Mean {"), key(EST, EST_SOURCE, "fit"), key(LIB, LIB_SOURCE, "helper")
        scale, impl = key(LIB, LIB_SOURCE, "SCALE"), key(EST, EST_SOURCE, "impl Mean")

        def item(k, mod, deps=(), bdeps=(), lookups=(), probes=()):
            return {"t": "item", "key": k, "kind": "Fn", "mod": mod, "deps": list(deps), "bdeps": list(bdeps),
                    "lookups": list(lookups), "probes": list(probes), "observes": observes, "includes": False}
        records = [
            {"t": "mod", "key": lib, "file": LIB, "chain": [], "parent": None, "name": ""},
            {"t": "mod", "key": est, "file": EST, "chain": [], "parent": lib, "name": "est"},
            {"t": "mod", "key": test, "file": TEST, "chain": [], "parent": None, "name": ""},
            {"t": "glob", "mod": lib, "src": est},
            item(key(TEST, TEST_SOURCE, "runs"), test, bdeps=[["full", mean], ["fn", fit]],
                 lookups=[[test, "Mean"]],
                 probes=[{"name": "fit", "early": True, "heads": [[mean, "Mean"]], "resolved": [], "traits": []}]),
            item(key(TEST, TEST_SOURCE, "c::Mean"), test, lookups=[[lib, "Mean"]]),
            item(fit, est, deps=[["head", impl]], bdeps=[["fn", helper], ["full", scale], ["full", mean]],
                 lookups=[[lib, "helper"], [lib, "SCALE"]]),
            item(impl, est, deps=[["full", mean]], lookups=[[est, "Mean"]]),
            item(helper, lib, lookups=[[lib, "f64"]]),
            item(scale, lib, lookups=[[lib, "f64"]]),
            item(mean, est, lookups=[[est, "f64"]]),
        ]
        return ei.Inventory.read(json.dumps(r) for r in records)

    def earn(self, observes=False, callee_bodies=False):
        selection = ei.Selection(self.root, self.inventory(observes), callee_bodies=callee_bodies)
        for rel, source, needle in ((TEST, TEST_SOURCE, "assert!"), (EST, EST_SOURCE, "self.bias +")):
            selection.execute(rel, source.index(needle))
        entries, tables = selection.entries(em.Checker(self.root, {}), ["crates/c"], "crates/c")
        for address, data in tables.items():
            self.write(f"{ei.STORE}/{address}.json", data.decode())
        return selection, entries

    def current(self, entries):
        checker = em.Checker(self.root, {})
        names = next(e for e in entries if e["kind"] == "rust_names")
        for entry in entries:
            if entry["kind"] == "rust_items" and not checker.items.items_current(entry, names):
                return False
            if entry["kind"] == "rust_guards" and not checker.items.guards_current(entry):
                return False
            if entry["kind"] == "evidence_file" and checker.input_digest("file", entry["path"]) != entry["digest"]:
                return False
        return True

    def test_selection_takes_executed_bodies_whole_and_called_functions_by_signature(self):
        selection, _ = self.earn()
        self.assertIn("full", selection.parts[EST]["impl:Mean#0/fn:fit#0"])
        self.assertEqual(selection.parts[LIB]["fn:helper#0"], {"sig"})
        self.assertEqual(selection.parts[LIB]["const:SCALE#0"], {"full"})
        self.assertIn("full", selection.parts[LIB]["mod:est#0"])
        self.assertNotIn("impl:Mean#0/fn:idle#0", selection.parts[EST])
        self.assertNotIn("fn:unused#0", selection.parts[LIB])

    def test_a_sampled_run_binds_called_functions_whole(self):
        # Coverage from a slice of the replicates: a function the executed code calls may
        # run in a replicate the slice did not, so its body is an input.
        selection, entries = self.earn(callee_bodies=True)
        self.assertIn("full", selection.parts[LIB]["fn:helper#0"])
        self.assertNotIn("impl:Mean#0/fn:idle#0", selection.parts[EST])  # never named: still out
        self.edit(LIB, "{ x + 1.0 }", "{ x + 2.0 }")
        self.assertFalse(self.current(entries))

    def test_unrelated_edits_leave_the_record_current(self):
        _, entries = self.earn()
        for rel, old, new in (
            (LIB, "{ x + 1.0 }", "{ x + 2.0 }"),                       # a called body that never ran
            (EST, "{ 0.0 }", "{ 1.0 }"),                               # an unexecuted method
            (LIB, "line!() }", "line!() + 1 }"),                       # an unexecuted position read
            (LIB, "pub const SCALE", "pub fn other() {}\n\npub const SCALE"),   # a new item
            ("crates/c/src/new.rs", None, "pub struct New;\n"),        # a new file
        ):
            with self.subTest(rel=rel, new=new):
                if old is None:
                    self.write(rel, new)
                else:
                    self.edit(rel, old, new)
                self.assertTrue(self.current(entries))

    def test_dependency_edits_owe_the_record(self):
        for rel, old, new in (
            (EST, "self.bias + crate", "self.bias * crate"),           # the executed body
            (LIB, "helper(x: f64) -> f64", "helper(x: f64) -> f32"),   # a called signature
            (LIB, "f64 = 2.0", "f64 = 3.0"),                           # a used constant
            (EST, "{ pub bias: f64 }", "{ pub bias: f64, pub n: u8 }"),  # a used type
            (EST, "    fn idle", "    pub fn fit(self) -> f64 { 0.0 }\n    fn idle"),  # an earlier method
            (LIB, "pub use est::*;", "pub use est::*;\npub struct Mean;"),  # a shadowing item
            (LIB, "pub mod est;", "#[cfg(any())]\npub mod est;"),       # the file leaves the crate
            (LIB, "#![allow(dead_code)]", "#![allow(unused)]"),         # a crate attribute
        ):
            with self.subTest(new=new):
                self.tearDown()
                self.setUp()
                _, entries = self.earn()
                self.edit(rel, old, new)
                self.assertFalse(self.current(entries))

    def test_a_position_observer_keeps_whole_files(self):
        selection, entries = self.earn(observes=True)
        self.assertIn(EST, selection.whole)
        self.edit(EST, "{ 0.0 }", "{ 1.0 }")
        self.assertFalse(self.current(entries))

    def test_an_unchanged_file_is_current_without_parsing(self):
        _, entries = self.earn()
        ei._PARSED.clear()
        self.assertTrue(self.current(entries))
        self.assertEqual(ei._PARSED, {})

    def test_names_filter_has_no_false_negatives(self):
        names = {f"name{i}" for i in range(500)}
        loaded = ei.Names.load(ei.Names.of(names).entry())
        self.assertTrue(all(n in loaded for n in names))


if __name__ == "__main__":
    unittest.main()
