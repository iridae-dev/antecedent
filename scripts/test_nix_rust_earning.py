"""Nix Rust earning orchestration with disposable, mocked test captures."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evidence_earning
import nix_evidence_build
import nix_evidence_capture
import nix_evidence_policy
import nix_evidence_source
import scientific_records as scientific


class NixRustEarning(unittest.TestCase):
    def test_incremental_targets_track_only_own_exact_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "parity/evidence_manifests").mkdir(parents=True)
            rows = [{"name": "a"}, {"name": "b"}]
            seen = []
            marks = []

            def locate(_, row):
                name = row["name"]
                return {"lang": "rust", "crate": name, "target": ["--lib"], "selector": name + "::lib::test"}

            def build(_, crate, kind, name, *, allowed_outputs):
                seen.append(dict(allowed_outputs))
                return {"source": {"commit": "head", "store_path": "/nix/store/source"}, "artifact": {"output": "/nix/store/" + crate}}

            class Snapshot:
                def __init__(self, _, row, binding, fp, *, nix_build, allowed_outputs):
                    self.name = row["name"]
                    self.manifest = {"record": {"fingerprint": "fp"}}

                def mark_nix_executed(self, binary):
                    marks.append(binary)

                def publish(self):
                    data = self.name.encode()
                    ref = hashlib.sha256(data).hexdigest()
                    (root / "parity/evidence_manifests" / (ref + ".json")).write_bytes(data)
                    return ref

            def ledger(*args, **kwargs):
                (root / "parity/scientific_records.toml").write_text("own ledger")
                return len(args[1])

            def capture(_, target_rows, built, run, **kwargs):
                return {"coverage": {row["name"] + "::lib::test": {} for row in target_rows}, "failures": []}

            with patch.object(scientific, "locate", side_effect=locate), \
                 patch.object(scientific, "Fingerprinter"), \
                 patch.object(scientific, "withdraw_records", side_effect=ledger), \
                 patch.object(scientific, "record_passes", side_effect=ledger) as record, \
                 patch.object(evidence_earning, "Snapshot", Snapshot), \
                 patch.object(evidence_earning, "project_snapshot"), \
                 patch.object(evidence_earning, "require_compile_environment"), \
                 patch.object(nix_evidence_build, "build_committed_target", side_effect=build), \
                 patch.object(nix_evidence_capture, "capture_built_target", side_effect=capture), \
                 patch.object(nix_evidence_policy, "validate_build"), \
                 patch.object(nix_evidence_source, "_clean_head", return_value=("head", "sha1")):
                self.assertEqual(scientific.earn_rust(root, rows, backend="nix"), 0)
            self.assertEqual(seen[0], {})
            self.assertIn("parity/scientific_records.toml", seen[1])
            self.assertIn("parity/evidence_manifests/" + hashlib.sha256(b"a").hexdigest() + ".json", seen[1])
            self.assertEqual(marks, ["/nix/store/a/bin/test", "/nix/store/b/bin/test"])
            self.assertEqual(record.call_args.kwargs["source_state"], ("head", False))

    def test_policy_failure_precedes_withdrawal_and_capture(self):
        info = {"lang": "rust", "crate": "a", "target": ["--lib"], "selector": "a::lib::test"}
        with patch.object(scientific, "locate", return_value=info), \
             patch.object(scientific, "Fingerprinter"), \
             patch.object(scientific, "withdraw_records") as withdraw, \
             patch.object(evidence_earning, "require_compile_environment"), \
             patch.object(nix_evidence_build, "build_committed_target", return_value={}), \
             patch.object(nix_evidence_capture, "capture_built_target") as capture, \
             patch.object(nix_evidence_policy, "validate_build", side_effect=RuntimeError("unreviewed")), \
             patch.object(nix_evidence_source, "_clean_head", return_value=("head", "sha1")):
            with self.assertRaisesRegex(RuntimeError, "unreviewed"):
                scientific.earn_rust(Path("/tmp"), [{}], backend="nix")
        withdraw.assert_not_called()
        capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
