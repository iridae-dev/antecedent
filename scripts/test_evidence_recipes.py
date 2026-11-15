"""Build recipes are bound by what they say: a comment or layout edit owes nothing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evidence_manifests as em
import evidence_recipes as recipes

NIX = """# The evidence build.
{ pkgs, src }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);  /* pinned */
  tools = import (builtins.fetchTree lock.nodes.nixpkgs.locked) { system = "aarch64-darwin"; };
in pkgs.runCommand "evidence" { } ''
  # a builder comment is part of the builder
  cargo test --no-run ${if src == null then "" else "--locked"}
''
"""
LOCK = {"nodes": {"nixpkgs": {"locked": {"narHash": "sha256-aaa", "rev": "1"}, "original": {"ref": "x"}},
                  "unused": {"locked": {"narHash": "sha256-bbb"}}, "root": {"inputs": {"nixpkgs": "nixpkgs"}}},
        "root": "root", "version": 7}
POLICY = '''"""Policy."""
# reviewed
POLICY = "audit-v1"


def allowed(path):
    """Whether a path is reviewed."""
    return path in ("/bin/sh",)
'''


class FormTests(unittest.TestCase):
    def test_nix_comments_and_layout_are_not_content(self):
        base = recipes.digest("nix", NIX.encode())
        edited = NIX.replace("# The evidence build.", "# Builds the evidence targets.\n#\n").replace(
            "/* pinned */", "").replace("{ pkgs, src }:", "{ pkgs\n, src\n}:")
        self.assertEqual(recipes.digest("nix", edited.encode()), base)

    def test_nix_code_and_string_contents_are_content(self):
        base = recipes.digest("nix", NIX.encode())
        for old, new in (('"aarch64-darwin"', '"x86_64-linux"'), ("--no-run", "--no-run --release"),
                         ("# a builder comment", "# another builder comment"), ("src == null", "src != null"),
                         ("pkgs.runCommand", "pkgs.runCommandLocal")):
            with self.subTest(old=old):
                self.assertNotEqual(recipes.digest("nix", NIX.replace(old, new).encode()), base)

    def test_a_string_is_never_taken_for_a_comment(self):
        tokens = recipes.nix_tokens('x = "a # b /* c"; y = \'\'\n  # kept ${"}"}\n\'\'; # dropped\n')
        self.assertIn('"a # b /* c"', tokens)
        self.assertTrue(any("# kept" in token for token in tokens))
        self.assertFalse(any("dropped" in token for token in tokens))
        with self.assertRaises(ValueError):
            recipes.nix_tokens('x = "unterminated')

    def test_a_lock_is_the_entries_the_recipes_read(self):
        self.assertEqual(recipes.lock_nodes([NIX]), ["nixpkgs"])
        base = recipes.digest("flake-lock", json.dumps(LOCK).encode(), nodes=["nixpkgs"])
        other = json.loads(json.dumps(LOCK))
        other["nodes"]["unused"]["locked"]["narHash"] = "sha256-ccc"
        other["nodes"]["nixpkgs"]["original"]["ref"] = "y"
        self.assertEqual(recipes.digest("flake-lock", json.dumps(other, indent=4).encode(), nodes=["nixpkgs"]), base)
        other["nodes"]["nixpkgs"]["locked"]["narHash"] = "sha256-ddd"
        self.assertNotEqual(recipes.digest("flake-lock", json.dumps(other).encode(), nodes=["nixpkgs"]), base)
        with self.assertRaises(ValueError):
            recipes.digest("flake-lock", json.dumps(LOCK).encode(), nodes=["missing"])

    def test_python_is_its_syntax_tree_without_documentation(self):
        base = recipes.digest("python", POLICY.encode())
        edited = POLICY.replace('"""Policy."""', '"""The reviewed policy.\n\nLonger."""').replace(
            "# reviewed", "# reviewed twice").replace('"""Whether a path is reviewed."""', '"""Reviewed?"""')
        self.assertEqual(recipes.digest("python", edited.encode()), base)
        self.assertNotEqual(recipes.digest("python", POLICY.replace('"audit-v1"', '"audit-v2"').encode()), base)
        self.assertNotEqual(recipes.digest("python", POLICY.replace('("/bin/sh",)', '("/bin/sh", "/tmp")').encode()), base)


class CurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        for rel, text in (("nix/evidence-target.nix", NIX), ("flake.lock", json.dumps(LOCK, indent=2)),
                          ("scripts/policy.py", POLICY)):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(text, encoding="utf-8")
        em._FILE_DIGESTS.clear()
        self.entries = [recipes.entry(rel, (self.root / rel).read_bytes(), "external", recipes=[NIX])
                        for rel in ("nix/evidence-target.nix", "flake.lock", "scripts/policy.py")]

    def tearDown(self):
        self.tmp.cleanup()

    def current(self):
        em._FILE_DIGESTS.clear()
        checker = em.Checker(self.root, {})
        return [checker.recipe_current(entry) for entry in self.entries]

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def test_an_entry_holds_the_bytes_it_was_read_from_and_its_content(self):
        nix, lock, policy = self.entries
        self.assertEqual((nix["form"], lock["form"], policy["form"]), ("nix", "flake-lock", "python"))
        self.assertEqual(lock["nodes"], ["nixpkgs"])
        self.assertEqual(nix["raw"], em.digest(NIX.encode()))
        self.assertIsNone(recipes.entry("rust-toolchain.toml", b"[toolchain]\n", "external"))

    def test_comment_and_layout_edits_leave_a_record_current(self):
        self.edit("nix/evidence-target.nix", "# The evidence build.", "# Reworded.")
        self.edit("flake.lock", '"version": 7', '"version":   7')
        self.edit("scripts/policy.py", "# reviewed", "# reviewed again")
        self.assertEqual(self.current(), [True, True, True])

    def test_a_change_to_what_is_built_owes_the_record(self):
        self.edit("nix/evidence-target.nix", "--no-run", "--no-run --release")
        self.edit("flake.lock", "sha256-aaa", "sha256-zzz")
        self.edit("scripts/policy.py", "audit-v1", "audit-v2")
        self.assertEqual(self.current(), [False, False, False])
        (self.root / "flake.lock").unlink()
        self.assertEqual(self.current()[1], False)


if __name__ == "__main__":
    unittest.main()
