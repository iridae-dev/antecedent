"""Identity mappings cannot confer authority before reviewed replay evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_key_aliases as aliases

ROOT = Path(__file__).resolve().parents[1]


class CalibrationIdentityAliasTests(unittest.TestCase):
    def pending_metadata(self) -> str:
        fields = {
            "record_id": aliases.RECORD_ID, "measurement_sha": aliases.MEASUREMENT_SHA,
            "collection_sha": aliases.COLLECTION_SHA, "original_record_sha256": aliases.RECORD_SHA256,
            "original_functional_sha256": aliases.FUNCTIONAL_SHA256,
            "status": "pending_identity_replay",
        }
        return "version = 1\n\n[[alias]]\n" + "".join(
            f"{key} = {json.dumps(value)}\n" for key, value in fields.items()
        )

    def test_pending_mapping_generates_no_runtime_authority(self) -> None:
        self.assertEqual(self.validate_modified(self.pending_metadata()), [])

    def validate_modified(self, metadata: str, records: str | None = None) -> list[dict[str, str]]:
        original = subprocess.run(
            ["git", "show", f"{aliases.COLLECTION_SHA}:parity/coverage_records.toml"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parity").mkdir()
            (root / "parity/calibration_key_aliases.toml").write_text(metadata, encoding="utf-8")
            (root / "parity/coverage_records.toml").write_text(
                records if records is not None else original, encoding="utf-8",
            )
            with patch.object(aliases.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=original,
            )):
                return aliases.validated_aliases(root)

    def test_status_flag_alone_cannot_activate_mapping(self) -> None:
        metadata = self.pending_metadata()
        with self.assertRaisesRegex(ValueError, "reviewed native construction"):
            self.validate_modified(metadata.replace("pending_identity_replay", "verified_identity_replay"))

    def test_receipt_hash_and_path_cannot_be_forged_or_escape_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parity").mkdir()
            raw = b'{"version":1}\n'
            (root / "parity/proof.json").write_bytes(raw)
            import hashlib
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(aliases.read_receipt(root, "parity/proof.json", digest), {"version": 1})
            for relative, expected_digest in (("parity/proof.json", "0" * 64),
                                               ("parity/../../outside.json", digest)):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    aliases.read_receipt(root, relative, expected_digest)

    def test_native_marker_requires_passing_original_test_and_one_object(self) -> None:
        valid = 'RECOVERY_IDENTITY_PROOF={"version":1}\ntest result: ok. 1 passed; 0 failed;\n'
        self.assertEqual(aliases.parse_native_log(valid), {"version": 1})
        for altered in (valid.replace("ok. 1 passed", "FAILED. 0 passed"),
                        valid + 'RECOVERY_IDENTITY_PROOF={}\n',
                        valid.replace('{"version":1}', '[]')):
            with self.subTest(log=altered), self.assertRaises(ValueError):
                aliases.parse_native_log(altered)

    def test_capture_refuses_linked_worktree_and_dirty_primary_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "primary 2.3.0"):
                aliases.clean_main_source(root)
            (root / ".git").unlink()
            (root / ".git").mkdir()
            with patch.object(aliases.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout="2.3.0\n"),
                subprocess.CompletedProcess([], 0, stdout=" M dirty.rs\n"),
            ]), self.assertRaisesRegex(ValueError, "clean committed main"):
                aliases.clean_main_source(root)

    def test_foreign_record_and_changed_measurement_sha_refuse(self) -> None:
        metadata = self.pending_metadata()
        for old, new in ((aliases.RECORD_ID, "cov.foreign"), (aliases.MEASUREMENT_SHA, "0" * 40)):
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, "exact original"):
                self.validate_modified(metadata.replace(old, new))

    def test_pending_mapping_cannot_embed_an_unverified_functional(self) -> None:
        metadata = self.pending_metadata()
        with self.assertRaisesRegex(ValueError, "unchecked runtime claims"):
            self.validate_modified(metadata + '\ncanonical_functional = "fabricated"\n')

    def test_verified_receipts_require_exact_native_protocol_and_all_replayed_values(self) -> None:
        import calibration_facets
        import collect_coverage_records

        records = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
        original = next(record for record in records["record"] if record["id"] == aliases.RECORD_ID)
        source = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
        identity = "recovery_scientific.v1:synthetic_test_construction"
        canonical = "recovered_effect:" + identity + ":treated=3ff0000000000000:control=0000000000000000"
        proof = {
            "version": 1, "record_id": aliases.RECORD_ID,
            "original_functional": original["functional"], "canonical_functional": canonical,
            "scientific_derivation_identity": identity,
            "rows": 1000, "input_digest": "a" * 32, "receipt_digest": "b" * 32,
            "identical_repeated_native_receipt": True,
            "negative_mutations": {"graph": "refused", "effect": "refused", "sampling": "different_scientific_identity"},
            "config": {"interval_method": "bootstrap_bca", "replicates": 2000,
                       "seed": 0x46ab0000 + 100000, "max_failed_fraction_bits": "0000000000000000",
                       "normalization_tolerance_bits": "3fb999999999999a", "small_cell_count_bits": "4014000000000000",
                       "treated_level_bits": "3ff0000000000000", "control_level_bits": "0000000000000000"},
        }
        def native_receipt(value: dict) -> dict:
            text = "RECOVERY_IDENTITY_PROOF=" + json.dumps(value) + "\ntest result: ok. 1 passed; 0 failed;\n"
            return {"version": 1, "source_sha": source, "branch": "2.3.0", "proof": value,
                    "log": text, "log_sha256": aliases.sha256(text)}
        raw = {key: value for key, value in original.items() if key not in calibration_facets.NOT_EMITTED}
        raw["functional"] = canonical
        text = "synthetic backend fixture log\n"
        replay = {"version": 1, "source_sha": source, "branch": "2.3.0", "record_id": aliases.RECORD_ID,
                  "logs": {"recovery.log": {"text": text, "sha256": aliases.sha256(text)}},
                  "raw_main_replay_record": raw}
        # Mock only the physical log backend. Scientific comparison runs the
        # production exact-float comparator; this fixture is never registry evidence.
        def check(native: dict, receipt: dict, emitted: dict) -> dict:
            with patch.object(collect_coverage_records, "log_problems", return_value=([], set())), \
                    patch.object(collect_coverage_records, "merged_records", return_value={aliases.RECORD_ID: emitted}):
                return aliases.validate_evidence(ROOT, original, native, receipt)
        self.assertEqual(check(native_receipt(proof), replay, raw)["source_sha"], source)
        for field in ("observed", "mcse", "n_min", "replicates_min", "estimator", "dependence", "grid"):
            changed = copy.deepcopy(raw)
            changed[field] = [] if field == "grid" else ("foreign" if isinstance(changed[field], str) else -1)
            changed_receipt = replay | {"raw_main_replay_record": changed}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "measured values or scope"):
                check(native_receipt(proof), changed_receipt, changed)
        for field in ("replicates", "interval_method", "treated_level_bits", "normalization_tolerance_bits"):
            changed = copy.deepcopy(proof)
            changed["config"][field] = "foreign"
            with self.subTest(config=field), self.assertRaisesRegex(ValueError, "original BCa"):
                check(native_receipt(changed), replay, raw)
        changed = copy.deepcopy(proof)
        changed["negative_mutations"]["graph"] = "same_identity"
        with self.assertRaisesRegex(ValueError, "original BCa"):
            check(native_receipt(changed), replay, raw)
        with self.assertRaisesRegex(ValueError, "main-branch source"):
            check(native_receipt(proof), replay | {"source_sha": "0" * 40}, raw)

    def test_original_record_values_axes_and_provenance_are_immutable(self) -> None:
        metadata = self.pending_metadata()
        records = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
        record = next(row for row in records["record"] if row["id"] == aliases.RECORD_ID)
        for field in ("observed", "estimator", "functional", "calibration_sha", "n_min"):
            changed = dict(record)
            changed[field] = "tampered" if isinstance(changed[field], str) else -1
            with self.subTest(field=field), patch.object(aliases.tomllib, "loads", side_effect=[
                tomllib.loads(metadata), {"record": [record]}, {"record": [changed]},
            ]), self.assertRaisesRegex(ValueError, "must remain unchanged"):
                self.validate_modified(metadata)


if __name__ == "__main__":
    unittest.main()
