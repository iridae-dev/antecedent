"""Engineering tests for the fail-closed manifest currency checker."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evidence_manifests as em


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("tools/evidence-analyzer/pin.toml",
                   'release = "1.85.1"\ncommit = "abc"\n')
        self.write("src/a.rs", "fn a() { 1; }\n")
        self.binding = {"kind": "scientific", "key": "crate::test", "fingerprint": "old", "measurement": None}

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
        return target

    def manifest(self, kind="file", key="src/a.rs"):
        checker = em.Checker(self.root, {})
        return {
            "schema": 1, "normalization": em.NORMALIZATION,
            "record": self.binding, "compiler": {"release": "1.85.1", "commit": "abc"},
            "analyzer": em.ANALYZER,
            "inputs": [{"kind": kind, "name" if kind in ("env", "build_env") else "path": key,
                        "digest": checker.input_digest(kind, key), "role": "selection"}],
            "fallbacks": [],
        }

    def store(self, manifest):
        ref = em.manifest_id(manifest)
        self.write(f"{em.STORE}/{ref}.json", json.dumps(manifest))
        return ref

    def check(self, manifest, binding=None):
        return em.Checker(self.root, {}).check(binding or self.binding, self.store(manifest))

    def test_a_matching_manifest_is_current(self):
        result = self.check(self.manifest())
        self.assertTrue(result["current"])
        self.assertEqual(result["reasons"], [])

    def test_body_edit_detected_even_with_preserved_timestamp_and_size(self):
        manifest = self.manifest()
        path = self.root / "src/a.rs"
        stat = path.stat()
        path.write_text("fn a() { 2; }\n", encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertIn("input_changed:file:src/a.rs", self.check(manifest)["reasons"])

    def test_directory_guard_detects_add_remove_rename_and_untracked_files(self):
        manifest = self.manifest("tree", "src")
        self.write("src/new.rs", "fn new() {}")
        self.assertFalse(self.check(manifest)["current"])
        (self.root / "src/new.rs").unlink()
        (self.root / "src/a.rs").rename(self.root / "src/b.rs")
        self.assertFalse(self.check(manifest)["current"])
        (self.root / "src/b.rs").unlink()
        self.assertFalse(self.check(manifest)["current"])

    def test_unrelated_file_outside_dependencies_does_not_change_snapshot(self):
        manifest = self.manifest()
        self.write("unrelated.rs", "fn unrelated() {}")
        self.assertTrue(self.check(manifest)["current"])

    def test_environment_distinguishes_unset_and_empty(self):
        manifest = self.manifest("env", "ANTECEDENT_EXAMPLE")
        ref = self.store(manifest)
        result = em.Checker(self.root, {"ANTECEDENT_EXAMPLE": ""}).check(self.binding, ref)
        self.assertFalse(result["current"])

    def test_missing_deleted_corrupt_or_misaddressed_manifest_owes(self):
        checker = em.Checker(self.root)
        for ref in (None, "../../escape", "a" * 64):
            self.assertFalse(checker.check(self.binding, ref)["current"])
        ref = self.store(self.manifest())
        path = self.root / em.STORE / f"{ref}.json"
        path.write_text("{", encoding="utf-8")
        self.assertFalse(em.Checker(self.root).check(self.binding, ref)["current"])
        path.write_text(json.dumps({"schema": 9}), encoding="utf-8")
        self.assertFalse(em.Checker(self.root).check(self.binding, ref)["current"])

    def test_deleted_input_owes(self):
        manifest = self.manifest()
        (self.root / "src/a.rs").unlink()
        self.assertFalse(self.check(manifest)["current"])

    def test_manifest_cannot_attest_another_record(self):
        manifest = self.manifest()
        self.assertFalse(self.check(manifest, {**self.binding, "key": "other"})["current"])
        self.assertFalse(self.check(manifest, {**self.binding, "fingerprint": "new"})["current"])

    def test_symlink_input_or_tree_member_owes(self):
        manifest = self.manifest("tree", "src")
        (self.root / "src/link").symlink_to(self.root / "src/a.rs")
        self.assertFalse(self.check(manifest)["current"])

    def test_identity_version_and_policy_fail_closed(self):
        for field, value in (("schema", 2), ("normalization", "unknown"),
                             ("compiler", {"release": "different", "commit": "abc"}),
                             ("analyzer", "different")):
            manifest = self.manifest()
            manifest[field] = value
            self.assertFalse(self.check(manifest)["current"], field)

    def test_no_unsupported_item_projection_silently_accepted(self):
        manifest = self.manifest()
        manifest["inputs"][0]["kind"] = "item"
        self.assertFalse(self.check(manifest)["current"])

    def test_guarded_file_selectively_tracks_bodies_but_guards_declarations(self):
        from evidence_projection import capture, fingerprint
        source = 'fn used()->i32 { 1 }\nfn unused()->i32 { 7 }\n'
        self.write('src/a.rs', source)
        start = source.index('{ 7 }')
        projection = capture(source, [(start, start + 5)])
        manifest = self.manifest()
        manifest['inputs'][0].update(kind='guarded_file', projection=projection,
                                     digest=fingerprint(projection))
        self.write('src/a.rs', source.replace('{ 7 }', '{ 8 }'))
        self.assertTrue(self.check(manifest)['current'])
        self.write('src/a.rs', source.replace('unused()', 'unused(x:i32)'))
        self.assertFalse(self.check(manifest)['current'])

    def test_guarded_file_cannot_hide_a_new_impl_in_a_body(self):
        from evidence_projection import capture, fingerprint
        source = 'fn unused() { 7 }'
        self.write('src/a.rs', source)
        projection = capture(source, [(source.index('{'), len(source))])
        manifest = self.manifest()
        manifest['inputs'][0].update(kind='guarded_file', projection=projection,
                                     digest=fingerprint(projection))
        self.write('src/a.rs', 'fn unused() { impl T for S {} 7 }')
        self.assertFalse(self.check(manifest)['current'])

    def test_fallback_must_hash_its_boundary(self):
        manifest = self.manifest()
        manifest["fallbacks"] = [{"path": "src", "reason": "opaque expansion"}]
        self.assertFalse(self.check(manifest)["current"])
        manifest = self.manifest("tree", "src")
        manifest["fallbacks"] = [{"path": "src", "reason": "opaque expansion"}]
        self.assertTrue(self.check(manifest)["current"])

    def test_deduplicates_shared_inputs_across_records_without_subprocess(self):
        manifest = self.manifest()
        ref = self.store(manifest)
        with patch("subprocess.run", side_effect=AssertionError("compiler invoked")), \
             patch("subprocess.check_output", side_effect=AssertionError("compiler invoked")):
            report = em.Checker(self.root, {}).report([(self.binding, ref)] * 1000)
        self.assertEqual(report["counts"]["current"], 1000)
        self.assertEqual(report["timings"]["files_hashed"], 1)

    def test_record_inventory_binds_probes_to_exact_measurement(self):
        self.write("parity/scientific_records.toml", '[[record]]\ntest="t"\nfingerprint="f"\n')
        self.write("parity/coverage_records.toml", '[[record]]\nid="c"\ncalibration_sha="m"\nfingerprint="f"\n')
        self.write("parity/calibration_probe_attestations.toml", '[[attestation]]\nrecord_id="c"\ncalibration_sha="m"\nsha="p"\nk=100\nbound=0.03\nresult="pass"\nfingerprint="f"\n')
        rows = em.inventory(self.root)
        self.assertEqual([binding["kind"] for binding, _ in rows], ["scientific", "calibration", "probe"])
        self.assertEqual(rows[-1][0]["measurement"], "m")
        self.assertEqual(rows[-1][0]["sha"], "p")
        self.assertEqual(rows[-1][0]["k"], 100)
        self.assertTrue(all(ref is None for _, ref in rows))

    def test_canonical_address_does_not_depend_on_object_key_order(self):
        manifest = self.manifest()
        self.assertEqual(em.manifest_id(manifest), em.manifest_id(dict(reversed(list(manifest.items())))))

    def test_unsafe_paths_are_rejected(self):
        for path in ("../outside", "/tmp/outside", "src/../a", "src//a", "src/./a", "src\\a"):
            with self.assertRaises(em.InvalidManifest, msg=path):
                em.relative_path(path)

    def test_the_checking_process_environment_is_never_an_input(self):
        ref = self.store(self.manifest("env", "ANTECEDENT_EXAMPLE"))
        build = self.manifest("build_env", "RUSTFLAGS")
        with patch.dict(os.environ, {"ANTECEDENT_EXAMPLE": "set", "RUSTFLAGS": "--cfg other"}):
            self.assertTrue(em.Checker(self.root).check(self.binding, ref)["current"])
            self.assertTrue(self.check(build)["current"])

    def test_lint_levels_are_not_build_configuration(self):
        self.assertEqual(em.noncanonical_environment({"RUSTFLAGS": "-D warnings -Wunused --deny=x"}, []), [])
        self.assertEqual(em.noncanonical_environment({"CARGO_ENCODED_RUSTFLAGS": "-D\x1fwarnings"}, []), [])
        self.assertTrue(em.noncanonical_environment({"RUSTFLAGS": "-D warnings -C target-cpu=native"}, []))
        self.assertTrue(em.noncanonical_environment({"CARGO_PROFILE_TEST_OPT_LEVEL": "3"}, []))
        self.assertEqual(em.noncanonical_environment({"ANTECEDENT_X": ""}, ["ANTECEDENT_X"]), ["ANTECEDENT_X=''"])

    def test_unknown_build_environment_input_is_rejected(self):
        manifest = self.manifest("build_env", "RUSTFLAGS")
        manifest["inputs"][0]["name"] = "PATH"
        self.assertFalse(self.check(manifest)["current"])

    def test_membership_in_a_git_work_tree_skips_ignored_files_only(self):
        git = lambda *args: subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)  # noqa: E731
        self.write(".gitignore", ".DS_Store\n*.so\n")
        git("init", "-q")
        git("add", "src/a.rs", ".gitignore")
        manifest = self.manifest("tree", "src")
        membership = self.manifest("membership", "src")
        for path in ("src/.DS_Store", "src/native.so"):
            self.write(path, "ignored host file or build product")
        self.assertTrue(self.check(manifest)["current"])
        self.assertTrue(self.check(membership)["current"])
        self.write("src/new.csv", "a runtime input added after earning")
        self.assertFalse(self.check(manifest)["current"])
        self.assertFalse(self.check(membership)["current"])


if __name__ == "__main__":
    unittest.main()
