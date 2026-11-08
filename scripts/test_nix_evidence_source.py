"""Disposable engineering checks for Nix source and coverage helpers."""
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nix_evidence_source import (
    import_committed_snapshot,
    map_coverage_sources,
    materialize_committed_source,
    verify_committed_snapshot,
)


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.test")
        self.git("config", "commit.gpgsign", "false")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "source.rs").write_text("fn main() {}\n", encoding="utf-8")
        self.git("add", "sub/source.rs")
        self.git("commit", "-qm", "fixture")
        self.destination = Path(self.tmp.name) / "snapshot"

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True).stdout

    def test_committed_tree_is_copied_with_its_commit_identity(self):
        result = materialize_committed_source(self.root, self.destination)
        self.assertEqual(result["commit"], self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual((self.destination / "sub/source.rs").read_bytes(), (self.root / "sub/source.rs").read_bytes())

    def test_dirty_or_untracked_source_refuses_without_creating_snapshot(self):
        (self.root / "untracked").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkout"):
            materialize_committed_source(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_only_exact_run_output_bytes_can_be_allowed(self):
        ledger = self.root / "parity/scientific_records.toml"
        ledger.parent.mkdir()
        ledger.write_text("run output", encoding="utf-8")
        allowed = {"parity/scientific_records.toml": hashlib.sha256(ledger.read_bytes()).hexdigest()}
        result = materialize_committed_source(self.root, self.destination, allowed_outputs=allowed)
        verify_committed_snapshot(self.root, self.destination, result["commit"], allowed_outputs=allowed)
        ledger.write_text("other edit", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkout"):
            verify_committed_snapshot(self.root, self.destination, result["commit"], allowed_outputs=allowed)

    def test_allowlist_cannot_whitelist_arbitrary_source_edits(self):
        path = self.root / "sub/source.rs"
        path.write_text("changed", encoding="utf-8")
        allowed = {"sub/source.rs": hashlib.sha256(path.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "checkout"):
            materialize_committed_source(self.root, self.destination, allowed_outputs=allowed)

    def test_export_substitution_and_omission_cannot_silently_change_snapshot(self):
        (self.root / ".gitattributes").write_text("sub/source.rs export-ignore\n", encoding="utf-8")
        self.git("add", ".gitattributes")
        self.git("commit", "-qm", "ignore from archive")
        with self.assertRaisesRegex(ValueError, "membership"):
            materialize_committed_source(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_symlinked_committed_input_is_refused(self):
        (self.root / "link").symlink_to("sub/source.rs")
        self.git("add", "link")
        self.git("commit", "-qm", "link")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            materialize_committed_source(self.root, self.destination)

    def test_verification_refuses_changed_snapshot_and_checkout(self):
        commit = materialize_committed_source(self.root, self.destination)["commit"]
        verify_committed_snapshot(self.root, self.destination, commit)
        path = self.destination / "sub/source.rs"
        path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_committed_snapshot(self.root, self.destination, commit)
        path.write_bytes((self.root / "sub/source.rs").read_bytes())
        (self.destination / "unexpected").mkdir()
        with self.assertRaisesRegex(ValueError, "membership"):
            verify_committed_snapshot(self.root, self.destination, commit)
        (self.destination / "unexpected").rmdir()
        (self.root / "sub/source.rs").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkout"):
            verify_committed_snapshot(self.root, self.destination, commit)

    def test_import_checks_git_and_nar_before_and_after(self):
        commit = materialize_committed_source(self.root, self.destination)["commit"]
        good_hash = "sha256-" + "A" * 43 + "="
        store = "/nix/store/" + "1" * 32 + "-antecedent-source"
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(stdout=store if "add-path" in command else good_hash)

        result = import_committed_snapshot(self.root, self.destination, commit, run=fake_run)
        self.assertEqual(result, {"commit": commit, "store_path": store, "nar_hash": good_hash})
        self.assertEqual(sum("hash" in command for command in commands), 3)
        self.assertTrue(all("--offline" in command for command in commands))

    def test_import_refuses_mismatched_store_and_mutated_snapshot(self):
        commit = materialize_committed_source(self.root, self.destination)["commit"]
        store = "/nix/store/" + "1" * 32 + "-antecedent-source"
        calls = 0

        def mismatched(command, **kwargs):
            nonlocal calls
            if "add-path" in command:
                return SimpleNamespace(stdout=store)
            calls += 1
            return SimpleNamespace(stdout="sha256-" + ("B" if calls == 2 else "A") * 43 + "=")

        with self.assertRaisesRegex(ValueError, "differs"):
            import_committed_snapshot(self.root, self.destination, commit, run=mismatched)

        def mutating(command, **kwargs):
            if "add-path" in command:
                (self.destination / "sub/source.rs").write_text("changed", encoding="utf-8")
                return SimpleNamespace(stdout=store)
            return SimpleNamespace(stdout="sha256-" + "A" * 43 + "=")

        with self.assertRaisesRegex(ValueError, "differs"):
            import_committed_snapshot(self.root, self.destination, commit, run=mutating)

    def test_import_rejects_invalid_store_path_and_nar_hash(self):
        commit = materialize_committed_source(self.root, self.destination)["commit"]
        good_hash = "sha256-" + "A" * 43 + "="
        def fake_run(command, **kwargs):
            return SimpleNamespace(stdout="/tmp/untrusted" if "add-path" in command else good_hash)
        with self.assertRaisesRegex(ValueError, "store path"):
            import_committed_snapshot(self.root, self.destination, commit, run=fake_run)
        with self.assertRaisesRegex(ValueError, "NAR hash"):
            import_committed_snapshot(self.root, self.destination, commit,
                                      run=lambda *a, **k: SimpleNamespace(stdout="fake"))

    def test_coverage_maps_only_matching_snapshot_sources(self):
        materialize_committed_source(self.root, self.destination)
        snapshot_file = self.destination / "sub/source.rs"
        checkout_file = self.root / "sub/source.rs"
        coverage = {str(snapshot_file): {3, 7}, "/external/dependency.rs": {2}}
        self.assertEqual(map_coverage_sources(coverage, self.destination, self.root), {str(checkout_file.resolve()): {3, 7}})
        checkout_file.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            map_coverage_sources(coverage, self.destination, self.root)

    def test_virtual_coverage_maps_only_verified_snapshot_bytes(self):
        materialize_committed_source(self.root, self.destination)
        target = self.root / "sub/source.rs"
        coverage = {"/antecedent-source/sub/source.rs": {2, 4}}
        self.assertEqual(map_coverage_sources(coverage, self.destination, self.root),
                         {str(target.resolve()): {2, 4}})
        with self.assertRaisesRegex(ValueError, "invalid"):
            map_coverage_sources({"/antecedent-source/../outside.rs": {1}}, self.destination, self.root)
        target.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            map_coverage_sources(coverage, self.destination, self.root)

    def test_coverage_does_not_follow_source_or_checkout_symlinks(self):
        materialize_committed_source(self.root, self.destination)
        snapshot_file = self.destination / "sub/source.rs"
        checkout_file = self.root / "sub/source.rs"
        checkout_file.unlink()
        checkout_file.symlink_to(snapshot_file)
        with self.assertRaisesRegex(ValueError, "symlink"):
            map_coverage_sources({str(snapshot_file): {1}}, self.destination, self.root)


if __name__ == "__main__":
    unittest.main()
