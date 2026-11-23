"""The manifest-only dependency source: manifests, lock and stubs, never real source."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nix_evidence_deps as deps


class DepsSourceTest(unittest.TestCase):
    def test_keeps_manifests_and_lock_and_stubs_every_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, tree = root / "snapshot", root / "deps"
            (snapshot / "crates/a/src").mkdir(parents=True)
            (snapshot / "crates/a/tests").mkdir()
            (snapshot / ".cargo").mkdir()
            (snapshot / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/a"]\nresolver = "2"\n', encoding="utf-8")
            (snapshot / ".cargo/config.toml").write_text("[resolver]\n", encoding="utf-8")
            (snapshot / "crates/a/Cargo.toml").write_text(
                '[package]\nname = "a"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8")
            (snapshot / "crates/a/src/lib.rs").write_text("pub fn secret() -> u8 { 7 }\n", encoding="utf-8")
            (snapshot / "crates/a/tests/t.rs").write_text("#[test]\nfn t() {}\n", encoding="utf-8")
            (snapshot / "crates/a/README.md").write_text("not a build input\n", encoding="utf-8")
            (snapshot / "Cargo.lock").write_text('version = 3\n\n[[package]]\nname = "a"\nversion = "0.1.0"\n',
                                                 encoding="utf-8")
            deps.write_deps_source(snapshot, tree)
            files = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
            self.assertEqual(files, [".cargo/config.toml", "Cargo.lock", "Cargo.toml", "crates/a/Cargo.toml",
                                     "crates/a/src/lib.rs", "crates/a/tests/t.rs"])
            # Source never enters the dependency build: a library is empty, a test a stub.
            self.assertEqual((tree / "crates/a/src/lib.rs").read_text(encoding="utf-8"), "")
            self.assertIn("fn main", (tree / "crates/a/tests/t.rs").read_text(encoding="utf-8"))
            # A source-only edit leaves the tree, and so the cached derivation, unchanged.
            (snapshot / "crates/a/src/lib.rs").write_text("pub fn secret() -> u8 { 8 }\n", encoding="utf-8")
            again = root / "again"
            deps.write_deps_source(snapshot, again)
            self.assertEqual({p.relative_to(again): p.read_bytes() for p in again.rglob("*") if p.is_file()},
                             {p.relative_to(tree): p.read_bytes() for p in tree.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
