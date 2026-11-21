"""Candidate citations use exact passing assertions from the prerequisite lanes."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_evidence as evidence


class ReleaseBuildCacheTest(unittest.TestCase):
    def test_bootstrapped_rustdoc_uses_a_separate_target_directory(self):
        import gate_public_api

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CARGO_TARGET_DIR": tmp}), \
             patch.object(gate_public_api.subprocess, "run") as cargo:
            directory = gate_public_api.run_rustdoc(["example"], None)
            env = cargo.call_args.kwargs["env"]
            self.assertEqual(env["RUSTC_BOOTSTRAP"], "1")
            self.assertEqual(env["CARGO_TARGET_DIR"], str(Path(tmp) / "public-api"))
            self.assertEqual(directory, Path(tmp) / "public-api/doc")
            explicit = Path(tmp) / "custom"
            self.assertEqual(gate_public_api.run_rustdoc(["example"], explicit), explicit / "doc")
            self.assertEqual(cargo.call_args.kwargs["env"]["CARGO_TARGET_DIR"], str(explicit))


class ReleaseCitationPassesTest(unittest.TestCase):
    def test_a_source_change_during_verification_fails_the_snapshot(self):
        from _source_cache import source_cache, verified_source_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claim"
            path.write_text("initial")
            with self.assertRaisesRegex(RuntimeError, "changed during verification"), verified_source_snapshot():
                self.assertEqual(source_cache(Path.read_text)(path), "initial")
                path.write_text("changed")

    def test_listing_removes_ignored_tests_and_bulk_collection_preserves_file_identity(self):
        def process(stdout):
            return subprocess.CompletedProcess([], 0, stdout, "")

        with patch.object(evidence.subprocess, "run", side_effect=[
            process("runs: test\nignored: test\n"), process("ignored: test\n"),
        ]):
            self.assertEqual(evidence.ReleasePasses.runnable_names("test-binary"), {"runs"})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "current.json").write_text("[]")
            passes = evidence.ReleasePasses(directory)
        with patch.object(evidence.subprocess, "run", return_value=process(
            "tests/test_one.py::TestGroup::test_case[1]\ntests/test_two.py::test_second\n"
        )):
            passes.collect_python()
        self.assertEqual(passes.python_nodes, {
            ("python/tests/test_one.py", "test_case"), ("python/tests/test_two.py", "test_second"),
        })

    def test_only_the_exact_passing_binary_and_module_can_supply_a_citation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "current.json").write_text(json.dumps([["python/tests/science.py", "truth"]]))
            (directory / "rust-1.xml").write_text(
                '<testsuites><testsuite name="pkg::suite">'
                '<testcase name="mod::works"/><testcase name="mod::ignored"><skipped/></testcase>'
                '<testcase name="mod::failed"><failure/></testcase>'
                '<testcase name="other::absent"/></testsuite></testsuites>'
            )
            passes = evidence.ReleasePasses(directory)
        rust = evidence.ROOT / "some.rs"
        with patch.object(evidence, "target_root", return_value=(rust, "pkg", ["--test", "suite"])), \
             patch.object(evidence, "libtest_name", side_effect=lambda path, name, root: "mod::" + name):
            self.assertTrue(passes.contains(rust, "works"))
            for name in ("failed", "ignored", "absent"):
                self.assertFalse(passes.contains(rust, name))
            with patch.object(evidence, "target_root", return_value=(rust, "pkg", ["--lib"])):
                self.assertFalse(passes.contains(rust, "works"))
        self.assertTrue(passes.contains(evidence.ROOT / "python/tests/science.py", "truth"))
        self.assertFalse(passes.contains(evidence.ROOT / "python/tests/other.py", "truth"))


if __name__ == "__main__":
    unittest.main()
