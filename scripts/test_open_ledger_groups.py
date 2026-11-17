#!/usr/bin/env python3
"""Open ledger rows name the gate groups that measure them, and the owed run executes those groups."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_groups as groups  # noqa: E402
import calibration_status as status  # noqa: E402
from _repo import REPO_ROOT, load_toml  # noqa: E402


def record(test: str) -> dict:
    return {"id": f"cov.{test}", "test": f"crates/antecedent/tests/{test}"}


class LedgerItemTests(unittest.TestCase):
    ROWS = [
        {"id": "pending:A", "groups": ["suite_a: first", "suite_a: second"]},
        {"id": "statement:r: s", "groups": ["suite_b: only"]},
        {"id": "cell:C", "groups": []},
    ]

    def test_a_row_owes_the_groups_that_have_no_record(self):
        records = [record("suite_a.rs::first"), record("suite_b.rs::only")]
        items = status.ledger_items(self.ROWS, records)
        self.assertEqual(
            [(i["id"], i["groups"], i["unwired"]) for i in items],
            [("pending:A", ["suite_a: second"], False), ("cell:C", [], True)],
        )
        self.assertTrue(all(i["kind"] == "ledger" and i["reason"] == "no_record" for i in items))

    def test_a_row_whose_groups_all_have_records_is_not_owed(self):
        records = [record("suite_a.rs::first"), record("suite_a.rs::second"), record("suite_b.rs::only")]
        self.assertEqual([i["id"] for i in status.ledger_items(self.ROWS[:2], records)], [])

    def test_a_row_naming_no_group_is_owed_and_unwired(self):
        (item,) = status.ledger_items([{"id": "pending:X", "groups": []}], [])
        self.assertEqual((item["groups"], item["unwired"]), ([], True))


class CandidateGroupTests(unittest.TestCase):
    GROUPS = [
        ("grid", "mediation_effect_pag_given_bayesian_coverage"),
        ("grid", "mediation_effect_pag_graph_posterior_bayesian_coverage"),
        ("grid", "mediation_effect_temporal_pag_graph_posterior_bayesian_coverage"),
    ]

    def test_a_graph_posterior_coordinate_is_not_measured_by_the_given_structure(self):
        key = ("MediationEffect", "Pag", "graph_posterior", "Bayesian")
        self.assertEqual(
            status.candidate_groups(key, self.GROUPS),
            ["grid: mediation_effect_pag_graph_posterior_bayesian_coverage"],
        )

    def test_a_given_coordinate_keeps_every_group_of_its_class(self):
        key = ("MediationEffect", "Pag", "fixed", "Bayesian")
        self.assertEqual(len(status.candidate_groups(key, self.GROUPS)), 2)


class OwedSelectionTests(unittest.TestCase):
    def test_an_open_ledger_row_selects_the_groups_it_names(self):
        labels = ["suite_new: brand_new", "suite_new: other", "suite_old: measured"]
        candidates = [groups.Group(i, label, False, True) for i, label in enumerate(labels, 1)]
        items = [
            {"kind": "ledger", "id": "pending:A", "reason": "no_record", "groups": ["suite_new: brand_new"]}
        ]
        facets = SimpleNamespace(
            current_owed=lambda: (items, {}, {}),
            load_records=lambda: [],
            split_group=lambda spec: tuple((spec.partition(": ")[0], spec.partition(": ")[2])),
        )
        with patch.dict(sys.modules, {"calibration_status": facets}):
            selection = groups.select_owed(candidates)
        self.assertEqual([g.index for g in selection.groups], [1])
        self.assertEqual(selection.reasons[1], "1 open ledger row(s) with no record")


class ShippedLedgerTests(unittest.TestCase):
    """The open ledgers of this tree: every row is wired to groups the owed run can execute."""

    @classmethod
    def setUpClass(cls):
        cls.rows = status.ledger_rows(REPO_ROOT)
        cls.table = load_toml(REPO_ROOT / "parity/calibration_groups.toml")["group"]

    def test_a_row_no_group_is_named_for_is_surfaced_as_an_unwired_owed_item(self):
        unwired = {row["id"] for row in self.rows if not row["groups"]}
        surfaced = {
            item["id"] for item in status.ledger_items(self.rows, []) if item["unwired"]
        }
        self.assertEqual(unwired, surfaced)

    def test_every_named_group_is_in_the_group_table(self):
        labels = {group["label"] for group in self.table}
        missing = [(row["id"], g) for row in self.rows for g in row["groups"] if g not in labels]
        self.assertEqual(missing, [])

    def test_every_named_group_runs_a_test_that_exists(self):
        by_label = {group["label"]: group for group in self.table}
        problems = []
        for row in self.rows:
            for label in row["groups"]:
                group = by_label.get(label)
                if group is None:
                    continue
                source = REPO_ROOT / "crates" / group["package"] / "tests" / f"{group['target']}.rs"
                if not source.is_file():
                    problems.append(f"{row['id']}: {label}: {source} does not exist")
                    continue
                text = source.read_text(encoding="utf-8")
                for name in group.get("tests", []):
                    function = str(name).rpartition("::")[2]
                    # A test is a `fn`, or a name a test macro of the file expands into one.
                    if not re.search(rf"\b{re.escape(function)}\b", text):
                        problems.append(f"{row['id']}: {label}: no test {function}")
        self.assertEqual(problems, [])

    def test_the_posterior_coordinates_the_readiness_report_lists_have_a_graph_posterior_group(self):
        groups_ = status.calibration_groups()
        for key in status.not_measured_coordinates():
            candidates = status.candidate_groups(key, groups_)
            self.assertTrue(candidates, f"{' / '.join(key)}: no candidate group")
            if key[2] == "graph_posterior":
                self.assertTrue(all("graph_posterior" in c for c in candidates), candidates)


if __name__ == "__main__":
    unittest.main()
