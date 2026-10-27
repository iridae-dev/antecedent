"""Source-frozen family provenance; no numerical calibration is executed."""

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_family as family


class FamilyTests(unittest.TestCase):
    def test_only_full_object_identities_cache_source_reads(self):
        family._immutable_source.cache_clear()
        self.addCleanup(family._immutable_source.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.rs"
            path.write_bytes(b"before")
            self.assertEqual(family.read_source(root, path.name), b"before")
            path.write_bytes(b"after")
            self.assertEqual(family.read_source(root, path.name), b"after")
            with patch.object(
                family.subprocess,
                "check_output",
                side_effect=[
                    b"fixed",
                    b"head1",
                    b"head2",
                    b"tag1",
                    b"tag2",
                    b"short1",
                    b"short2",
                ],
            ) as show:
                for _ in range(2):
                    self.assertEqual(
                        family.read_source(root, path.name, "a" * 40), b"fixed"
                    )
                self.assertEqual(show.call_count, 1)
                for ref, expected in (
                    ("HEAD", b"head"),
                    ("v1", b"tag"),
                    ("abcdef", b"short"),
                ):
                    self.assertEqual(
                        family.read_source(root, path.name, ref), expected + b"1"
                    )
                    self.assertEqual(
                        family.read_source(root, path.name, ref), expected + b"2"
                    )
                self.assertEqual(show.call_count, 7)

    def test_roster_ignores_inherited_selection_and_needs_only_historical_inputs(self):
        gate = b"""#!/usr/bin/env bash
set -eu
[ "$ANTECEDENT_CALIBRATION_DRY_RUN" = 1 ]
[ -z "${ANTECEDENT_CALIBRATION_SHARD:-}" ]
[ -z "${ANTECEDENT_CALIBRATION_DECISION_CONFIG:-}" ]
printf 'group 1: alpha\ngroup 2: beta\ngrid 2\n'
"""
        with patch.dict(
            os.environ,
            {
                "ANTECEDENT_CALIBRATION_SHARD": "9/10",
                "ANTECEDENT_CALIBRATION_DECISION_CONFIG": "invalid",
            },
        ):
            self.assertEqual(
                family.gate_roster(gate, b""),
                [
                    {"index": 1, "label": "alpha", "grid": False},
                    {"index": 2, "label": "beta", "grid": True},
                ],
            )

    def test_duplicate_or_partial_roster_is_rejected(self):
        for output in (
            "group 2: alpha",
            "group 1: alpha\\ngroup 2: alpha",
            "group 1: a:b\\ngroup 2: a/b",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                family.gate_roster(f"printf '{output}\\n'".encode(), b"")

    def test_lexical_constructor_audit_preserves_literals_and_nested_arguments(self):
        text = """// CoverageTally::new("fake", 0.1)
let x = "CoverageTally::new(fake)";
CoverageTally::for_record(key.with_dgp("a(b)"), level(0.95)).labelled("a\\\"b");
"""
        rows = family.producer_calls(text, labels=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["argument"], 'key.with_dgp("a(b)"), level(0.95)')
        self.assertEqual(len(family.labelled_calls(text)), 1)

    def test_new_unlabelled_producer_requires_explicit_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "crates").mkdir()
            (root / "crates/new.rs").write_text(
                'CoverageTally::new("case", 0.95);', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "explicit source-fixed-family review"):
                family.reviewed_label_audit(root)

    def test_review_is_invalidated_by_constructor_level_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "crates").mkdir()
            path = root / "crates/knn_matching_calibration.rs"
            path.write_text('CoverageTally::new("case", 0.95);', encoding="utf-8")
            audit = family.reviewed_label_audit(root)
            family.validate_label_audit(audit, root)
            path.write_text('CoverageTally::new("case", 0.90);', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                family.validate_label_audit(audit, root)

    def test_historical_domain_is_bound_to_measured_literal_not_current_default(self):
        old = family.decision.LEGACY_DOMAIN
        with patch.object(family, "read_source", return_value=f'DOMAIN = "{old}"\n'.encode()):
            self.assertEqual(family.source_validation_domain(sha="a" * 40), old)
        for source in (
            b'DOMAIN = "unregistered"',
            b"DOMAIN = choose_domain()",
            b'DOMAIN = "release_2_4_1_validation_v1"\nDOMAIN = "release_2_4_1_validation_v1"',
            b'# DOMAIN = "release_2_4_1_validation_v1"',
        ):
            with (
                patch.object(family, "read_source", return_value=source),
                self.assertRaises(ValueError),
            ):
                family.source_validation_domain(sha="a" * 40)

    def test_old_manifest_keeps_old_domain_without_resetting_risk(self):
        manifest = {"groups": [{"index": 1}], "validation_domain": family.decision.LEGACY_DOMAIN}
        old = family.decision.config_for_manifest(manifest)
        new = family.decision.config_for_manifest(
            {**manifest, "validation_domain": family.decision.DOMAIN}
        )
        self.assertNotEqual(old["validation_domain"], new["validation_domain"])
        for field in ("family_cells", "family_alpha", "cell_cap_per_group", "looks"):
            self.assertEqual(old[field], new[field])
        with self.assertRaises(ValueError):
            family.decision.config_for_manifest({**manifest, "validation_domain": "arbitrary"})

    def test_cumulative_slots_preserve_original_risk_and_reject_reuse(self):
        allocations = family.allocations
        receipt = json.loads((family.ROOT / allocations.PATH).read_text(encoding="utf-8"))
        original = json.loads(
            family.read_source(family.ROOT, family.MANIFEST_PATH, allocations.ORIGINAL_SOURCE)
        )
        groups = original["groups"]
        allocations.validate(receipt, groups)
        variants = []
        for field, value in (
            ("slot", 1),
            ("look_weight", 0.5),
            ("planned_replicates", 400),
            ("se_kind", "hc1_sampling_weighted"),
            ("functional", allocations.FUNCTIONAL),
        ):
            changed = copy.deepcopy(receipt)
            changed["allocations"][1][field] = value
            variants.append(changed)
        changed = copy.deepcopy(receipt)
        changed["allocations"].append(copy.deepcopy(changed["allocations"][1]))
        variants.append(changed)
        changed = copy.deepcopy(receipt)
        changed["original_family"]["family_alpha"] = 0.1
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                allocations.validate(changed, groups)
        with self.assertRaisesRegex(ValueError, "original roster"):
            allocations.validate(receipt, groups[:-1])

    def test_followup_selection_allows_only_registered_group_and_construction(self):
        groups = json.loads(
            family.read_source(
                family.ROOT, family.MANIFEST_PATH, family.allocations.ORIGINAL_SOURCE
            )
        )["groups"]
        for domain in (
            family.decision.HC3_DOMAIN,
            family.decision.PAG_DOMAIN,
            family.decision.PAG_CONFIDENCE_SET_DOMAIN,
        ):
            config = family.decision.config_for_manifest(
                {"groups": groups, "validation_domain": domain}
            )
            label, expected = family.followup_allocation(config)
            family.validate_measurement_selection(config, [label])
            for selection in (["unallocated: test"], [g["label"] for g in groups]):
                with (
                    self.subTest(domain=domain),
                    self.assertRaisesRegex(ValueError, "allocated group"),
                ):
                    family.validate_measurement_selection(config, selection)
            record = {
                k: expected[k] for k in ("functional", "interval_method", "se_kind", "nominal")
            }
            record["id"] = expected["record_id"]
            family.validate_measurement_construction(config, label, record)
            with self.assertRaisesRegex(ValueError, "unallocated construction"):
                family.validate_measurement_construction(
                    config, label, {**record, "functional": "old"}
                )
        legacy = family.decision.config_for_manifest(
            {"groups": groups, "validation_domain": family.decision.LEGACY_DOMAIN}
        )
        family.validate_measurement_selection(legacy, [g["label"] for g in groups])
        with self.assertRaisesRegex(ValueError, "original roster"):
            family.validate_measurement_selection(legacy, ["future: foreign"])
        hc3 = family.decision.config_for_manifest(
            {"groups": groups, "validation_domain": family.decision.HC3_DOMAIN}
        )
        with (
            patch.dict(os.environ, {"ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL": "2"}),
            self.assertRaisesRegex(ValueError, "protocol3"),
        ):
            family.validate_measurement_selection(hc3, [family.allocations.GROUP])

    def test_actual_gate_dispatch_rejects_foreign_followup_before_any_command(self):
        groups = json.loads(
            family.read_source(
                family.ROOT, family.MANIFEST_PATH, family.allocations.ORIGINAL_SOURCE
            )
        )["groups"]
        config = family.decision.config_for_manifest(
            {"groups": groups, "validation_domain": family.decision.PAG_CONFIDENCE_SET_DOMAIN}
        )
        label, _ = family.followup_allocation(config)
        gate = (family.ROOT / family.GATE_PATH).read_text(encoding="utf-8")
        check = "check() {" + gate.split("\ncheck() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        with tempfile.TemporaryDirectory() as directory:
            script = "set -e\nGROUP_INDEX=0\nSHARD_N=''\nGRID_POINTS='0 1 2'\nCONFIRM_SUFFIX=''\nFAILED_COUNT=0\n"
            script += (
                "grid_group() { return 0; }\nrun_point() { echo DISPATCH; RUN_STATUS=0; }\n" + check
            )
            environment = {
                **os.environ,
                "ANTECEDENT_CALIBRATION_DRY_RUN": "",
                family.decision.ENV: family.decision.canonical_json(config),
                "LOG_DIR": directory,
            }
            denied = subprocess.run(
                ["bash", "-c", script + "check 'foreign: test' true\n"],
                cwd=family.ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("allocated group", denied.stderr)
            self.assertNotIn("DISPATCH", denied.stdout)
            allowed = subprocess.run(
                ["bash", "-c", script + f"check '{label}' true\n"],
                cwd=family.ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertEqual(allowed.stdout.count("DISPATCH"), 3)

    def test_current_domain_is_resolvable_by_the_immutable_source_reader(self):
        source = (family.ROOT / "scripts/calibration_decision.py").read_bytes()
        with patch.object(family, "read_source", return_value=source):
            self.assertEqual(
                family.source_validation_domain(family.ROOT, "0" * 40),
                family.decision.PAG_CONFIDENCE_SET_DOMAIN,
            )

    def test_pag_confidence_set_reserves_distinct_slot_without_resetting_spent_risk(self):
        allocations = family.allocations
        receipt = json.loads((family.ROOT / allocations.PATH).read_text(encoding="utf-8"))
        groups = json.loads(
            family.read_source(family.ROOT, family.MANIFEST_PATH, allocations.ORIGINAL_SOURCE)
        )["groups"]
        domain = family.decision.PAG_CONFIDENCE_SET_DOMAIN
        allocations.validate(receipt, groups, domain)
        historical = copy.deepcopy(receipt)
        historical.pop("pag_confidence_set_extension")
        allocations.validate(historical, groups, family.decision.PAG_DOMAIN)
        with self.assertRaisesRegex(ValueError, "all cumulative"):
            allocations.validate(historical, groups, domain)
        for key, value in (
            ("slot", 2),
            ("planned_replicates", 2500),
            ("look_weight", 0.5),
            ("validation_domain", family.decision.PAG_DOMAIN),
        ):
            changed = copy.deepcopy(receipt)
            changed["pag_confidence_set_extension"]["new_slot"][key] = value
            with self.assertRaisesRegex(ValueError, "confidence-set slot"):
                allocations.validate(changed, groups, domain)
        changed = copy.deepcopy(receipt)
        changed["pag_confidence_set_extension"]["prior_slot"]["raw_log_sha256_by_point"]["0"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "cumulative spending"):
            allocations.validate(changed, groups, domain)

    def test_pag_successor_preserves_hc3_history_and_rejects_slot_reuse(self):
        allocations = family.allocations
        receipt = json.loads((family.ROOT / allocations.PATH).read_text(encoding="utf-8"))
        groups = json.loads(
            family.read_source(family.ROOT, family.MANIFEST_PATH, allocations.ORIGINAL_SOURCE)
        )["groups"]
        historical = copy.deepcopy(receipt)
        historical.pop("pag_trial_shift_extension")
        allocations.validate(historical, groups, family.decision.HC3_DOMAIN)
        with self.assertRaisesRegex(ValueError, "prospective cumulative"):
            allocations.validate(historical, groups, family.decision.PAG_DOMAIN)
        allocations.validate(receipt, groups, family.decision.PAG_DOMAIN)
        for key, value in (
            ("slot", 1),
            ("planned_replicates", 20000),
            ("validation_domain", family.decision.LEGACY_DOMAIN),
        ):
            changed = copy.deepcopy(receipt)
            changed["pag_trial_shift_extension"]["new_slot"][key] = value
            with self.assertRaisesRegex(ValueError, "PAG cumulative"):
                allocations.validate(changed, groups, family.decision.PAG_DOMAIN)
        old_hc3 = family.config_at_sha("da811c715e4d1961ca0cba7356467e34f2ec25e1")
        self.assertEqual(old_hc3["validation_domain"], family.decision.HC3_DOMAIN)
        self.assertEqual(
            old_hc3["family_digest"],
            "806de9511ace7382c6be49d81f4822ad24a58f27d3c59a6dd58adcc62c58ca0a",
        )

    def test_manifest_binds_complete_roster_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("crates", "scripts", "parity"):
                (root / name).mkdir()
            (root / family.GATE_PATH).write_text(
                "printf 'group 1: alpha\\ngroup 2: beta\\ngrid 2\\n'", encoding="utf-8"
            )
            (root / family.ACCEPTANCE_PATH).write_text("", encoding="utf-8")
            audit = family.reviewed_label_audit(root)
            with patch.object(family, "allocation_digest", return_value="a" * 64):
                manifest = family.build_manifest(root, audit)
                config = family.validate_manifest(manifest, audit, root)
            self.assertEqual(config["family_cells"], 2 * 3 * 256)
            manifest["groups"] = manifest["groups"][:1]
            with (
                patch.object(family, "allocation_digest", return_value="a" * 64),
                self.assertRaisesRegex(ValueError, "complete frozen source roster"),
            ):
                family.validate_manifest(manifest, audit, root)


if __name__ == "__main__":
    unittest.main()
