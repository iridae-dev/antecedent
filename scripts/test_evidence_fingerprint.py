#!/usr/bin/env python3
"""The shared test fingerprint: one implementation for scientific and calibration records."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import evidence_fingerprint as ef  # noqa: E402
import scientific_records as sr  # noqa: E402


def package(root: Path, name: str, deps=()):
    return {
        "name": name,
        "manifest_path": str(root / "crates" / name / "Cargo.toml"),
        "dependencies": [{"name": d, "kind": None} for d in deps],
    }


class SharedFingerprintTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.metadata = {"packages": [package(self.root, "leaf"), package(self.root, "top", ["leaf"])]}
        for name in ("leaf", "top"):
            self.write(f"crates/{name}/Cargo.toml", f'[package]\nname = "{name}"\nversion = "1.0.0"\n')
            self.write(f"crates/{name}/src/lib.rs", f"pub fn {name}() -> u8 {{ 1 }}\n")
        self.write("crates/top/tests/cov.rs", "#[test]\nfn cov_case() {}\n")
        self.write("crates/top/tests/dgp.rs", "pub fn dgp() {}\n")
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/*"]\n')
        self.write("Cargo.lock", "version = 4\n")
        self.write("rust-toolchain.toml", '[toolchain]\nchannel = "1.90"\n')

    def write(self, rel: str, text: str):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def record(self, **extra):
        return {
            "id": "cov.x",
            "test": "crates/top/tests/cov.rs::cov_case",
            "dgp": "crates/top/tests/dgp.rs::dgp",
            **extra,
        }

    def fingerprint(self, record=None, platform=True):
        fp = ef.Fingerprinter(self.root, self.metadata, "", platform=platform)
        return ef.coverage_fingerprint(fp, record or self.record())

    def test_scientific_records_use_the_same_fingerprinter(self):
        self.assertIs(sr.Fingerprinter, ef.Fingerprinter)
        self.assertIs(sr.locate, ef.locate)

    def test_a_coverage_record_fingerprints_like_the_scientific_row_of_its_test(self):
        fp = ef.Fingerprinter(self.root, self.metadata, "")
        row = {
            "evidence_test": "crates/top/tests/cov.rs",
            "evidence_assertion": "cov_case",
            "fixtures": ["crates/top/tests/dgp.rs"],
        }
        self.assertEqual(self.fingerprint(), fp.fingerprint(row))

    def test_coverage_row_splits_the_selector_and_keeps_a_foreign_dgp_as_a_fixture(self):
        row = ef.coverage_row("crates/a/tests/t.rs::name", "crates/a/tests/t.rs::dgp")
        self.assertEqual((row["evidence_test"], row["evidence_assertion"], row["fixtures"]), ("crates/a/tests/t.rs", "name", []))
        self.assertEqual(ef.coverage_row("crates/a/tests/t.rs::n", "crates/b/tests/d.rs::g")["fixtures"], ["crates/b/tests/d.rs"])

    def test_dependency_dgp_and_test_edits_change_it_and_comments_do_not(self):
        before = self.fingerprint()
        self.write("crates/top/tests/cov.rs", "// note\n#[test]\nfn cov_case() {}\n")
        self.assertEqual(before, self.fingerprint())
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 2 }\n")
        after_leaf = self.fingerprint()
        self.assertNotEqual(before, after_leaf)
        self.write("crates/top/tests/dgp.rs", "pub fn dgp() { let _ = 1; }\n")
        self.assertNotEqual(after_leaf, self.fingerprint())

    def test_platform_can_be_left_out(self):
        self.assertEqual(self.fingerprint(platform=False), self.fingerprint(platform=False))
        self.assertNotEqual(self.fingerprint(platform=False), self.fingerprint(platform=True))

    def test_an_unlocatable_test_has_no_fingerprint(self):
        self.assertIsNone(self.fingerprint(self.record(test="not-a-test-path::x")))


if __name__ == "__main__":
    unittest.main()
