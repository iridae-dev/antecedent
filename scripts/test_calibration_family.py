"""Source-frozen family provenance; no numerical calibration is executed."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_family as family


class FamilyTests(unittest.TestCase):
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
