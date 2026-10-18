#!/usr/bin/env python3
"""Independent boundaries for the one reviewed metadata-only capture transfer."""

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _temporal_capture_transfer as transfer
import calibration_facets as facets

ROOT = Path(__file__).resolve().parents[1]


def source(ref):
    return subprocess.check_output(["git", "show", f"{ref}:{transfer.PATH}"], cwd=ROOT, text=True)


class TemporalTransfer(unittest.TestCase):
    def test_exact_sources_and_entire_compilation_input_tree(self):
        old, new = source(transfer.ORIGINAL_SOURCE), source(transfer.CORRECTION_SOURCE)
        self.assertIsNone(transfer.prove_sources(transfer.PATH, old, new))
        proof = transfer.verify_input_transfer(ROOT, transfer.CORRECTION_SOURCE)
        self.assertEqual(proof["changed_inputs"], [transfer.PATH])
        self.assertGreater(proof["compiled_input_entries"], 1000)

    def test_source_body_literals_unknown_fields_and_wrong_paths_fail(self):
        old, new = source(transfer.ORIGINAL_SOURCE), source(transfer.CORRECTION_SOURCE)
        for altered in (
            new + "\n",
            new.replace('se_kind: "percentile"', 'se_kind: "normal"'),
            new.replace("grid_n(4000)", "grid_n(4001)"),
        ):
            self.assertIsNotNone(transfer.prove_sources(transfer.PATH, old, altered))
        self.assertIsNotNone(transfer.prove_sources("other.rs", old, new))
        self.assertIsNotNone(transfer.prove_sources(transfer.PATH, new, old))

    def test_other_compiled_input_or_missing_input_fails(self):
        for before, after in (
            (
                {transfer.PATH: "old", "Cargo.toml": "a"},
                {transfer.PATH: "new", "Cargo.toml": "b"},
            ),
            ({transfer.PATH: "old"}, {transfer.PATH: "new", "crates/other.rs": "x"}),
        ):
            with (
                patch.object(transfer, "compiled_inputs", side_effect=[before, after]),
                self.assertRaises(ValueError),
            ):
                transfer.verify_input_transfer(ROOT, transfer.CORRECTION_SOURCE)

    def payload(self):
        return {
            transfer.RECORD: {
                "id": transfer.RECORD,
                "se_kind": transfer.OLD_KIND,
                "interval_method": "block_percentile_bootstrap",
                "graph_class": "TemporalCpdag",
                "dependence": "block",
                "observed": 0.95,
                "replicates": 400,
                "width": -0.0,
            }
        }

    def test_projection_changes_exactly_one_field_and_preserves_original(self):
        payload = self.payload()
        before = copy.deepcopy(payload)
        derived = transfer.derive_payloads(payload, [transfer.RECORD])
        self.assertEqual(payload, before)
        self.assertEqual(derived[transfer.RECORD]["se_kind"], "percentile")
        self.assertEqual(
            facets.compare_replay(derived[transfer.RECORD], payload[transfer.RECORD]),
            [
                f"{transfer.RECORD}: se_kind stored 'percentile' replayed 'percentile_or_logical_support'"
            ],
        )
        self.assertTrue(
            facets._bit_equal(
                derived[transfer.RECORD] | {"se_kind": transfer.OLD_KIND},
                before[transfer.RECORD],
            )
        )

    def test_other_identity_method_or_already_corrected_kind_fails(self):
        for key, value in (
            ("se_kind", "percentile"),
            ("interval_method", "percentile_bootstrap"),
            ("graph_class", "Pag"),
            ("dependence", "iid"),
            ("id", "other"),
        ):
            payload = self.payload()
            payload[transfer.RECORD][key] = value
            with self.assertRaises(ValueError):
                transfer.derive_payloads(payload, [transfer.RECORD])
        for ids in ([], ["other"], [transfer.RECORD, "other"]):
            with self.assertRaises(ValueError):
                transfer.derive_payloads(self.payload(), ids)

    def test_composition_requires_exact_children_sources_selection_and_numeric_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            binary = "/actual/final-binary"
            gate = b"original gate"
            digest = hashlib.sha256(gate).hexdigest()
            build = {
                "source": "M",
                "purpose": "final_source",
                "default_features": True,
                "status": "passed",
                "executables": {binary: {"sha256": "binary-hash"}},
            }
            raw = json.dumps(build).encode()
            (path / "build.json").write_bytes(raw)
            (path / "ordinary.log").write_text(
                binary + "\ntest result: ok. 1 passed;\n", encoding="utf-8"
            )
            payloads = {
                record: {"id": record, "observed": 0.95, "width": 0.125}
                for record in transfer.ORDINARY_RECORDS
            }
            transferred = {transfer.RECORD: {"id": transfer.RECORD, "se_kind": "percentile"}}
            capture = {
                "version": 3,
                "kind": "temporal_and_final_source_composition",
                "source_sha": "M",
                "records": [transfer.RECORD, *transfer.ORDINARY_RECORDS],
                "temporal_capture": {"original_capture": {"logs": {"temporal.log": "oldhash"}}},
                "ordinary_capture": {
                    "version": 1,
                    "source_sha": "M",
                    "records": list(transfer.ORDINARY_RECORDS),
                    "logs": {
                        "ordinary.log": hashlib.sha256(
                            (path / "ordinary.log").read_bytes()
                        ).hexdigest()
                    },
                    "payloads": payloads,
                },
                "ordinary_execution": {
                    "source": "M",
                    "gate": {"before": digest, "after": digest},
                    "build_receipt": "build.json",
                    "build_receipt_sha256": hashlib.sha256(raw).hexdigest(),
                    "binaries": {binary: {"before": "binary-hash", "after": "binary-hash"}},
                },
                "payloads": transferred | payloads,
            }

            def load(value):
                with (
                    patch.object(transfer, "load_transfer", return_value=transferred),
                    patch.object(transfer, "_git", return_value=gate),
                ):
                    return transfer.load_composite(
                        path,
                        value,
                        "M",
                        [transfer.RECORD, *transfer.ORDINARY_RECORDS],
                        ROOT,
                        lambda paths: payloads,
                        facets._bit_equal,
                    )

            self.assertEqual(load(capture), capture["payloads"])
            wrong_features = copy.deepcopy(build)
            wrong_features["default_features"] = False
            wrong_raw = json.dumps(wrong_features).encode()
            (path / "build.json").write_bytes(wrong_raw)
            altered = copy.deepcopy(capture)
            altered["ordinary_execution"]["build_receipt_sha256"] = hashlib.sha256(
                wrong_raw
            ).hexdigest()
            with self.assertRaises(ValueError):
                load(altered)
            (path / "build.json").write_bytes(raw)
            original_log = (path / "ordinary.log").read_bytes()
            wrong_log = original_log + b'calibration-record {"measured_at":"old-source"}\n'
            (path / "ordinary.log").write_bytes(wrong_log)
            altered = copy.deepcopy(capture)
            altered["ordinary_capture"]["logs"]["ordinary.log"] = hashlib.sha256(
                wrong_log
            ).hexdigest()
            with self.assertRaises(ValueError):
                load(altered)
            (path / "ordinary.log").write_bytes(original_log)
            mutations = [
                lambda c: c.update(source_sha="wrong"),
                lambda c: c.pop("temporal_capture"),
                lambda c: c["ordinary_capture"].update(source_sha="old"),
                lambda c: c["ordinary_capture"]["records"].append(transfer.RECORD),
                lambda c: c["records"].append("extra"),
                lambda c: c["ordinary_execution"]["gate"].update(after="wrong"),
                lambda c: c["ordinary_execution"]["binaries"][binary].update(after="wrong"),
                lambda c: c["payloads"][transfer.ORDINARY_RECORDS[0]].update(width=9.0),
            ]
            for mutation in mutations:
                altered = copy.deepcopy(capture)
                mutation(altered)
                with self.assertRaises(ValueError):
                    load(altered)
            altered = copy.deepcopy(capture)
            altered["ordinary_capture"]["logs"] = {"temporal.log": "oldhash"}
            with self.assertRaises(ValueError):
                load(altered)

    def test_execution_proof_requires_guarded_binary_and_all_original_terminal_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            build = {
                "source": transfer.ORIGINAL_SOURCE,
                "status": "passed",
                "executables": {
                    transfer.ORIGINAL_BINARY: {"sha256": transfer.ORIGINAL_BINARY_SHA256}
                },
            }
            receipt = json.dumps(build).encode()
            (path / "original-build-receipt.json").write_bytes(receipt)
            receipt_hash = hashlib.sha256(receipt).hexdigest()
            execution = {
                "source": transfer.ORIGINAL_SOURCE,
                "build_receipt_sha256": receipt_hash,
                "gate_sha256": transfer.ORIGINAL_GATE_SHA256,
                "binary": transfer.ORIGINAL_BINARY,
                "binary_sha256_before": transfer.ORIGINAL_BINARY_SHA256,
                "binary_sha256_after": transfer.ORIGINAL_BINARY_SHA256,
            }
            logs = {}
            for point, budget in ((0, 400), (1, 2000), (2, 400)):
                name = f"p{point}.log"
                value = {
                    "id": transfer.RECORD,
                    "measured_at": transfer.ORIGINAL_SOURCE,
                    "grid_point": point,
                    "decision_protocol": {"planned_replicates": budget},
                }
                (path / name).write_text(
                    transfer.ORIGINAL_BINARY
                    + "\ncalibration-record "
                    + json.dumps(value)
                    + "\ntest result: ok. 1 passed;\n",
                    encoding="utf-8",
                )
                logs[name] = "hash-checked-by-loader"
            with patch.object(transfer, "ORIGINAL_BUILD_SHA256", receipt_hash):
                transfer.verify_execution(path, {"original_execution": execution}, logs)
                for field in ("source", "gate_sha256", "binary_sha256_after"):
                    with self.assertRaises(ValueError):
                        transfer.verify_execution(
                            path, {"original_execution": execution | {field: "wrong"}}, logs
                        )
                with self.assertRaises(ValueError):
                    transfer.verify_execution(
                        path, {"original_execution": execution}, {"p0.log": "x"}
                    )
                original = (path / "p1.log").read_text(encoding="utf-8")
                for text in (
                    original.replace("2000", "400"),
                    original.replace("1 passed", "0 passed"),
                    original.replace(transfer.ORIGINAL_BINARY, "/other"),
                    original.replace(transfer.ORIGINAL_SOURCE, "other"),
                ):
                    (path / "p1.log").write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        transfer.verify_execution(path, {"original_execution": execution}, logs)

    def test_derived_capture_retains_verified_raw_source_and_cannot_hide_numeric_edits(
        self,
    ):
        payloads = self.payload()
        proof = {"reviewed": "exact"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = b"immutable numerical output\n"
            (path / "original.log").write_bytes(raw)
            capture = {
                "version": 2,
                "kind": "temporal_construction_metadata_transfer",
                "source_sha": "F",
                "records": [transfer.RECORD],
                "source_transfer": proof,
                "original_capture": {
                    "version": 1,
                    "source_sha": transfer.ORIGINAL_SOURCE,
                    "records": [transfer.RECORD],
                    "logs": {"original.log": hashlib.sha256(raw).hexdigest()},
                    "payloads": payloads,
                },
                "payloads": transfer.derive_payloads(payloads, [transfer.RECORD]),
            }

            def load(value):
                with (
                    patch.object(transfer, "verify_input_transfer", return_value=proof),
                    patch.object(transfer, "verify_execution"),
                ):
                    return transfer.load_transfer(
                        path,
                        value,
                        "F",
                        [transfer.RECORD],
                        ROOT,
                        lambda paths: payloads,
                        facets._bit_equal,
                    )

            self.assertEqual(load(capture), capture["payloads"])
            for key in ("observed", "replicates", "width"):
                altered = copy.deepcopy(capture)
                altered["payloads"][transfer.RECORD][key] = 1
                with self.assertRaises(ValueError):
                    load(altered)
            for mutation in (
                lambda c: c.update(source_sha="other"),
                lambda c: c["original_capture"].update(source_sha="F"),
                lambda c: c.update(source_transfer={}),
                lambda c: c["original_capture"]["logs"].update({"../escape": "x"}),
            ):
                altered = copy.deepcopy(capture)
                mutation(altered)
                with self.assertRaises(ValueError):
                    load(altered)
            (path / "original.log").write_bytes(b"altered")
            with self.assertRaises(ValueError):
                load(capture)


if __name__ == "__main__":
    unittest.main()
