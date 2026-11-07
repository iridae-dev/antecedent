"""Mocked Nix capture checks: no scientific assertion is executed."""
import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

import nix_evidence_capture as nix_capture
import scientific_records


class NixCaptureTest(unittest.TestCase):
    root = Path("/tmp/nix-capture-fixture")
    row = {"evidence_test": "crates/toy/src/lib.rs", "evidence_assertion": "toy"}
    info = {"lang": "rust", "selector": "toy::lib::toy", "crate": "toy", "target": ["--lib"]}
    source = {"commit": "abc", "store_path": "/nix/store/source", "nar_hash": "nar"}
    artifact = {"source": "/nix/store/source", "source_nar_hash": "nar", "output": "/nix/store/output"}
    built = {"source": source, "artifact": artifact, "sandbox_verified": False, "evidence_authorized": False}

    def test_nix_gate_stops_before_discovery_build_or_ledger_changes(self):
        with patch.object(scientific_records, "all_scientific_rows") as rows, \
             patch.object(scientific_records, "withdraw_records") as withdraw, \
             patch.object(scientific_records, "record_passes") as record:
            self.assertEqual(scientific_records.cmd_run(self.root, argparse.Namespace(rust_build_backend="nix")), 1)
        rows.assert_not_called()
        withdraw.assert_not_called()
        record.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            scientific_records.earn_rust(self.root, [self.row], backend="nix")

    def test_capture_uses_exact_artifact_binary_and_rechecks_before_after(self):
        capture = {"/nix/store/source/crates/toy/src/lib.rs": {1}}
        mapped = {"/tmp/nix-capture-fixture/crates/toy/src/lib.rs": {1}}
        with patch.object(scientific_records, "locate", return_value=self.info), \
             patch.object(scientific_records, "capture_env", return_value={}), \
             patch.object(scientific_records, "capture_one", return_value=capture) as execute, \
             patch.object(nix_capture, "build_committed_target", return_value=self.built), \
             patch.object(nix_capture, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(nix_capture, "import_committed_snapshot", return_value=self.source), \
             patch.object(nix_capture, "verify_committed_snapshot") as verify, \
             patch.object(nix_capture, "recheck_artifact") as recheck, \
             patch.object(nix_capture, "map_coverage_sources", return_value=mapped) as remap:
            result = nix_capture.diagnose_target(self.root, [self.row])
        self.assertEqual(execute.call_args.args[1], "/nix/store/output/bin/test")
        self.assertEqual(execute.call_args.kwargs["llvm_tools"], {
            "llvm-profdata": "/nix/store/output/bin/llvm-profdata", "llvm-cov": "/nix/store/output/bin/llvm-cov"})
        self.assertEqual(result["coverage"][self.info["selector"]], mapped)
        self.assertFalse(result["sandbox_verified"])
        self.assertFalse(result["evidence_authorized"])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(recheck.call_count, 2)
        remap.assert_called_once()

    def test_capture_rechecks_on_exception_and_never_publishes(self):
        with patch.object(scientific_records, "locate", return_value=self.info), \
             patch.object(scientific_records, "capture_env", return_value={}), \
             patch.object(scientific_records, "capture_one", side_effect=RuntimeError("fixture failure")), \
             patch.object(scientific_records, "record_passes") as record, \
             patch.object(nix_capture, "build_committed_target", return_value=self.built), \
             patch.object(nix_capture, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(nix_capture, "import_committed_snapshot", return_value=self.source), \
             patch.object(nix_capture, "verify_committed_snapshot") as verify, \
             patch.object(nix_capture, "recheck_artifact") as recheck:
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                nix_capture.diagnose_target(self.root, [self.row])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(recheck.call_count, 2)
        record.assert_not_called()

    def test_unexpected_evidence_authorization_is_refused(self):
        built = {**self.built, "evidence_authorized": True}
        with patch.object(scientific_records, "locate", return_value=self.info), \
             patch.object(scientific_records, "capture_one") as execute, \
             patch.object(nix_capture, "build_committed_target", return_value=built):
            with self.assertRaisesRegex(ValueError, "authorize"):
                nix_capture.diagnose_target(self.root, [self.row])
        execute.assert_not_called()

    def test_python_and_mixed_targets_refuse_before_build(self):
        python = {"lang": "python", "crate": None, "target": [], "selector": "test_a.py::test_a"}
        other = {"lang": "rust", "crate": "other", "target": ["--lib"], "selector": "other::lib::toy"}
        with patch.object(scientific_records, "locate", return_value=python), \
             patch.object(nix_capture, "build_committed_target") as build:
            with self.assertRaisesRegex(ValueError, "Python"):
                nix_capture.diagnose_target(self.root, [self.row])
            build.assert_not_called()
        with patch.object(scientific_records, "locate", side_effect=[self.info, other]), \
             patch.object(nix_capture, "build_committed_target") as build:
            with self.assertRaisesRegex(ValueError, "one Cargo target"):
                nix_capture.diagnose_target(self.root, [self.row, self.row])
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
