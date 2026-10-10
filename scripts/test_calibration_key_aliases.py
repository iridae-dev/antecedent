"""Identity mappings cannot confer authority before reviewed replay evidence."""

from __future__ import annotations

import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_key_aliases as aliases

ROOT = Path(__file__).resolve().parents[1]


class CalibrationIdentityAliasTests(unittest.TestCase):
    def test_pending_mapping_generates_no_runtime_authority(self) -> None:
        self.assertEqual(aliases.validated_aliases(ROOT), [])

    def validate_modified(self, metadata: str, records: str | None = None) -> None:
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
                aliases.validated_aliases(root)

    def test_status_flag_alone_cannot_activate_mapping(self) -> None:
        metadata = (ROOT / "parity/calibration_key_aliases.toml").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reviewed native construction"):
            self.validate_modified(metadata.replace("pending_identity_replay", "verified_identity_replay"))

    def test_foreign_record_and_changed_measurement_sha_refuse(self) -> None:
        metadata = (ROOT / "parity/calibration_key_aliases.toml").read_text(encoding="utf-8")
        for old, new in ((aliases.RECORD_ID, "cov.foreign"), (aliases.MEASUREMENT_SHA, "0" * 40)):
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, "exact original"):
                self.validate_modified(metadata.replace(old, new))

    def test_pending_mapping_cannot_embed_an_unverified_functional(self) -> None:
        metadata = (ROOT / "parity/calibration_key_aliases.toml").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unchecked runtime claims"):
            self.validate_modified(metadata + '\ncanonical_functional = "fabricated"\n')

    def test_original_record_values_axes_and_provenance_are_immutable(self) -> None:
        metadata = (ROOT / "parity/calibration_key_aliases.toml").read_text(encoding="utf-8")
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
