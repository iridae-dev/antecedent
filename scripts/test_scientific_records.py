#!/usr/bin/env python3
"""Scientific record fingerprints: what a change invalidates and what it leaves current."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import scientific_records as sr  # noqa: E402

LOCK = """version = 4

[[package]]
name = "antecedent-leaf"
version = "1.0.0"

[[package]]
name = "antecedent-mid"
version = "1.0.0"
dependencies = ["antecedent-leaf"]

[[package]]
name = "serde"
version = "1.0.0"
"""


def package(root: Path, name: str, deps=(), dev=()):
    return {
        "name": name,
        "manifest_path": str(root / "crates" / name / "Cargo.toml"),
        "dependencies": [{"name": d, "kind": None} for d in deps]
        + [{"name": d, "kind": "dev"} for d in dev]
        + [{"name": "serde", "kind": None}],
    }


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.metadata = {
            "packages": [
                package(self.root, "leaf"),
                package(self.root, "mid", deps=["leaf"]),
                package(self.root, "top", deps=["mid"]),
                package(self.root, "other"),
            ]
        }
        for name in ("leaf", "mid", "top", "other"):
            self.write(f"crates/{name}/Cargo.toml", f'[package]\nname = "{name}"\nversion = "1.0.0"\n')
            self.write(f"crates/{name}/src/lib.rs", f"pub fn {name}() -> u8 {{ 1 }}\n")
            self.write(f"crates/{name}/tests/t.rs", f"#[test]\nfn check_{name}() {{}}\n")
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/*"]\n[workspace.package]\nversion = "1.0.0"\n')
        self.write("Cargo.lock", LOCK)
        self.write("rust-toolchain.toml", '[toolchain]\nchannel = "1.90"\n')
        self.write("conformance/a/expected.json", '{"x": 1}')
        self.write("conformance/b/expected.json", '{"x": 2}')

    def write(self, rel: str, text: str):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def row(self, crate: str, fixtures=()):
        return {
            "evidence_test": f"crates/{crate}/tests/t.rs",
            "evidence_assertion": f"check_{crate}",
            "roles": ["software", "scientific"],
            "fixtures": list(fixtures),
        }

    def fps(self, features: str = ""):
        fp = sr.Fingerprinter(self.root, self.metadata, features)
        return {
            c: fp.fingerprint(self.row(c, ["conformance/a"] if c == "mid" else ["conformance/b"] if c == "top" else []))
            for c in ("leaf", "mid", "top", "other")
        }


class FingerprintTests(Fixture):
    def changed(self, before, after):
        return {k for k in before if before[k] != after[k]}

    def test_unchanged_inputs_are_stable(self):
        self.assertEqual(self.fps(), self.fps())

    def test_comment_and_blank_edits_do_not_invalidate(self):
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", "// note\n\npub fn leaf() -> u8 { 1 } // trailing\n")
        self.assertEqual(before, self.fps())

    def test_literal_edit_invalidates(self):
        self.write("crates/leaf/src/lib.rs", 'pub fn leaf() -> &\'static str { "a" }\n')
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", 'pub fn leaf() -> &\'static str { "b" }\n')
        self.assertTrue(self.changed(before, self.fps()))

    def test_leaf_source_change_owes_only_dependents(self):
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 2 }\n")
        self.assertEqual(self.changed(before, self.fps()), {"leaf", "mid", "top"})

    def test_middle_change_does_not_owe_dependency(self):
        before = self.fps()
        self.write("crates/mid/src/lib.rs", "pub fn mid() -> u8 { 2 }\n")
        self.assertEqual(self.changed(before, self.fps()), {"mid", "top"})

    def test_fixture_change_owes_only_citers(self):
        before = self.fps()
        self.write("conformance/a/expected.json", '{"x": 99}')
        self.assertEqual(self.changed(before, self.fps()), {"mid"})

    def test_test_file_change_owes_only_that_test(self):
        before = self.fps()
        self.write("crates/top/tests/t.rs", "#[test]\nfn check_top() { assert!(true); }\n")
        self.assertEqual(self.changed(before, self.fps()), {"top"})

    def test_lock_version_only_bump_stays_current(self):
        before = self.fps()
        self.write("Cargo.lock", LOCK.replace('version = "1.0.0"\n\n[[package]]\nname = "antecedent-mid"',
                                              'version = "1.1.0"\n\n[[package]]\nname = "antecedent-mid"'))
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/*"]\n[workspace.package]\nversion = "1.1.0"\n')
        self.assertEqual(before, self.fps())

    def test_external_dependency_change_invalidates_all(self):
        before = self.fps()
        self.write("Cargo.lock", LOCK.replace('name = "serde"\nversion = "1.0.0"', 'name = "serde"\nversion = "1.0.1"'))
        self.assertEqual(self.changed(before, self.fps()), set(before))

    def test_feature_change_invalidates_all(self):
        self.assertEqual(self.changed(self.fps(), self.fps("--features x")), {"leaf", "mid", "top", "other"})

    def test_toolchain_change_invalidates_all(self):
        before = self.fps()
        self.write("rust-toolchain.toml", '[toolchain]\nchannel = "1.91"\n')
        self.assertEqual(self.changed(before, self.fps()), set(before))

    def test_dev_dependency_counts_for_the_test_crate_only(self):
        self.metadata["packages"][3] = package(self.root, "other", dev=["leaf"])
        self.metadata["packages"][2] = package(self.root, "top", deps=["other"])
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 3 }\n")
        after = self.fps()
        self.assertIn("other", self.changed(before, after))
        self.assertNotIn("top", self.changed(before, after))


class RecordTests(Fixture):
    def test_current_owed_and_roundtrip(self):
        fp = sr.Fingerprinter(self.root, self.metadata, "")
        rows = [self.row("leaf"), self.row("top")]
        owed, current = sr.classify(self.root, rows, fp)
        self.assertEqual((len(owed), len(current)), (2, 0))
        sr.write_records(self.root, {})
        sr.record_passes(self.root, owed[:1])
        owed, current = sr.classify(self.root, rows, fp)
        self.assertEqual([o["selector"] for o in owed], ["top::t::check_top"])
        self.assertEqual([c["selector"] for c in current], ["leaf::t::check_leaf"])
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 5 }\n")
        owed, current = sr.classify(self.root, rows, sr.Fingerprinter(self.root, self.metadata, ""))
        self.assertEqual(len(owed), 2)

    def test_native_closure_covers_python_rows(self):
        self.metadata["packages"].append({
            "name": "native", "manifest_path": str(self.root / "python/Cargo.toml"),
            "dependencies": [{"name": "leaf", "kind": None}],
        })
        self.write("python/Cargo.toml", '[package]\nname = "native"\nversion = "1.0.0"\n')
        self.write("python/src/lib.rs", "pub fn n() {}\n")
        self.write("python/antecedent/__init__.py", "x = 1\n")
        self.write("python/tests/test_a.py", "def test_a(): pass\n")
        row = {"evidence_test": "python/tests/test_a.py", "evidence_assertion": "test_a", "roles": ["scientific"], "fixtures": []}
        fp1 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 9 }\n")
        fp2 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.write("python/antecedent/__init__.py", "x = 2\n")
        fp3 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.write("crates/other/src/lib.rs", "pub fn other() -> u8 { 9 }\n")
        fp4 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.assertEqual(len({fp1, fp2, fp3}), 3)
        self.assertEqual(fp3, fp4)
        self.assertEqual(sr.locate(self.root, row)["selector"], "tests/test_a.py::test_a")


if __name__ == "__main__":
    unittest.main()
