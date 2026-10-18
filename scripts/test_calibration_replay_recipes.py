"""Original attempted budgets survive current acceptance-planner changes."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import calibration_facets as facets
from _calibration_replay_recipes import from_registry, replay_environments

ROOT = Path(__file__).resolve().parent.parent


class OriginalReplayRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(
            (ROOT / "parity/calibration_original_replay_recipes.json").read_text(encoding="utf-8")
        )
        cls.records = {r["id"]: r for r in facets.load_records()}

    def test_every_carried_original_recipe_matches_its_actual_stored_payload(self) -> None:
        self.assertEqual(len(self.registry["records"]), 32)
        for entry in self.registry["records"]:
            record = self.records[entry["record_id"]]
            with self.subTest(record=record["id"]):
                result = replay_environments([record], self.registry)
                self.assertEqual(
                    [int(env["ANTECEDENT_CALIBRATION_NSIM"]) for env in result],
                    [point["planned_attempts"] for point in entry["points"]],
                )
                for env in result:
                    self.assertEqual(
                        env["ANTECEDENT_CALIBRATION_ACCEPTANCE_NSIM"],
                        env["ANTECEDENT_CALIBRATION_NSIM"],
                    )

    def test_actual_producer_blobs_resolve_at_each_historical_source(self) -> None:
        for entry in self.registry["records"]:
            self.assertTrue(from_registry([self.records[entry["record_id"]]], ROOT))

    def test_parent_legacy_middle400_is_preserved_despite_current_minimum2000(self) -> None:
        entry = next(
            row
            for row in self.registry["records"]
            if "parent_set_knowledge" in row["record_id"] and row["source_sha"].startswith("c30")
        )
        result = replay_environments([self.records[entry["record_id"]]], self.registry)
        self.assertEqual(
            [int(env["ANTECEDENT_CALIBRATION_NSIM"]) for env in result], [2000, 400, 400]
        )

    def test_source_or_payload_change_cannot_reuse_a_recipe(self) -> None:
        record = self.records[self.registry["records"][0]["record_id"]]
        for key, value in (("calibration_sha", "0" * 40), ("nominal", 0.99)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "source/payload"):
                replay_environments([{**record, key: value}], self.registry)
        tagged = {**record, "facets": ["retagged"], "surface_list_blob": "0" * 40}
        self.assertEqual(
            replay_environments([tagged], self.registry),
            replay_environments([record], self.registry),
        )

    def test_missing_hash_unknown_budget_or_changed_prefix_is_rejected(self) -> None:
        record = self.records[self.registry["records"][0]["record_id"]]
        for key, value in (
            ("log_sha256", ""),
            ("producer_blob", ""),
            ("planned_attempts", 1000),
            ("computed_indices", [400, 2000]),
            ("replicates", 399),
            ("domain", "new"),
        ):
            registry = copy.deepcopy(self.registry)
            registry["records"][0]["points"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay_environments([record], registry)

    def test_partial_group_and_different_budgets_fail_closed(self) -> None:
        record = self.records[self.registry["records"][0]["record_id"]]
        with self.assertRaisesRegex(ValueError, "separate groups"):
            replay_environments([record, {"id": "unknown"}], self.registry)
        other = next(
            self.records[row["record_id"]]
            for row in self.registry["records"]
            if row["points"][1]["planned_attempts"] != 400
        )
        with self.assertRaisesRegex(ValueError, "separate groups"):
            replay_environments([record, other], self.registry)

    def test_unregistered_records_keep_existing_protocol_selection(self) -> None:
        self.assertEqual(replay_environments([{"id": "other"}], self.registry), [])


if __name__ == "__main__":
    unittest.main()
