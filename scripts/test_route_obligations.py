#!/usr/bin/env python3
"""Route obligations: generation from the recipes and the checker's four failures."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import check_route_obligations as chk  # noqa: E402
import generate_route_obligations as gen  # noqa: E402

AXES = {
    "query": ["AverageEffect", "Counterfactual"],
    "graph_class": ["Dag", "Admg"],
    "structure": ["explicit"],
    "inference": ["Frequentist"],
    "validation": ["none"],
}
COORD = dict(
    query="AverageEffect",
    graph_class="Dag",
    structure="explicit",
    inference="Frequentist",
    validation="none",
)
CLOSED = [
    {
        "queries": ["Counterfactual"],
        "graph_classes": ["Admg"],
        "reason": "needs a supplied Dag",
        "executes_unlicensed": True,
    }
]
RECORD = "cov.example.record"


def sources() -> dict:
    cell = dict(
        COORD,
        estimators=["linear.adjustment.ate"],
        checked_execution=[
            {
                "estimator": "linear.adjustment.ate",
                "test": "t/example.rs",
                "assertion": "retains_the_plan",
            }
        ],
        evidence_kind="internal_known_truth",
        evidence_test="t/example.rs",
        evidence_assertion="known_truth",
        known_truth_fixture="conformance/example",
        calibration=[RECORD],
        calibration_reason="boundary_record",
    )
    matrix_cell = dict(
        COORD, route_estimator="linear.adjustment.ate", estimators=["linear.adjustment.ate"]
    )
    return {
        "licensed": [cell],
        "closed": CLOSED,
        "na": [],
        "axes": AXES,
        "matrix": {"licensed": [matrix_cell], "closed_rules": CLOSED, "na_rules": []},
    }


def run(rows, src=None, cite=None, exists=None, ids=None) -> list[str]:
    return chk.check(
        rows,
        src or sources(),
        cite=cite or (lambda t, a: []),
        fixture_exists=exists or (lambda p: True),
        coverage_ids=ids if ids is not None else {RECORD},
    )


def rows_of(src=None) -> list[dict]:
    return copy.deepcopy(gen.build(src or sources()))


class RouteObligationTest(unittest.TestCase):
    def test_generation_emits_one_row_per_executing_route(self) -> None:
        rows = rows_of()
        self.assertEqual([r["kind"] for r in rows], ["licensed", "execution_only"])
        licensed, closed = rows
        self.assertEqual(
            licensed["compiler_execution"],
            ["linear.adjustment.ate = t/example.rs::retains_the_plan"],
        )
        self.assertEqual(
            licensed["identification_license"],
            "internal_known_truth|t/example.rs|known_truth|conformance/example",
        )
        self.assertEqual(licensed["calibration_license"], f"records:{RECORD}")
        self.assertEqual(closed["coordinates"], 1)
        self.assertEqual(closed["identification_license"], "none")
        self.assertEqual(run(rows), [])

    def test_a_missing_obligation_fails(self) -> None:
        problems = run(rows_of()[1:])
        self.assertTrue(any(p.startswith("missing:") and "AverageEffect" in p for p in problems))

    def test_a_row_stale_against_the_matrix_fails(self) -> None:
        rows = rows_of()
        src = sources()
        src["matrix"]["licensed"][0]["route_estimator"] = "other.estimator"
        problems = run(rows, src)
        self.assertTrue(any(p.startswith("stale:") and "matrix_digest" in p for p in problems))
        extra = rows_of()
        extra.append(dict(extra[0], id="Counterfactual:Dag:explicit:Frequentist:none"))
        self.assertTrue(any("has no executing route" in p for p in run(extra)))

    def test_an_unreachable_citation_fails(self) -> None:

        def bad_cite(t: str, a: str) -> list[str]:
            return [f"test {t!r} does not exist"]

        self.assertTrue(any(p.startswith("unreachable:") for p in run(rows_of(), cite=bad_cite)))
        self.assertTrue(
            any(
                "conformance/example" in p
                for p in run(rows_of(), exists=lambda p: p != "conformance/example")
            )
        )
        self.assertTrue(any("coverage record" in p for p in run(rows_of(), ids=set())))
        missing = chk.check(
            rows_of(),
            sources(),
            assertion_present=lambda rel, name: [f"{rel} does not name {name!r}"],
        )
        self.assertTrue(any("does not name 'known_truth'" in p for p in missing))

    def test_a_route_licensed_only_by_a_compatible_recipe_fails(self) -> None:
        rows = rows_of()
        rows[0]["license_basis"] = "recipe_compatible"
        self.assertTrue(any(p.startswith("recipe_only:") and "basis" in p for p in run(rows)))
        src = sources()
        for name in ("evidence_kind", "calibration", "calibration_reason"):
            src["licensed"][0].pop(name)
        stripped = rows_of(src)
        problems = run(stripped, src)
        self.assertTrue(any("identification license" in p for p in problems))
        self.assertTrue(any("calibration license" in p for p in problems))
        closed = rows_of()
        closed[1]["calibration_license"] = "reason:no_interval_reported"
        self.assertTrue(any("executes unlicensed yet carries" in p for p in run(closed)))

    def test_a_closed_route_cannot_be_licensed_by_its_recipe(self) -> None:
        src = sources()
        src["licensed"].append(dict(src["licensed"][0], query="Counterfactual", graph_class="Admg"))
        src["matrix"]["licensed"].append(
            dict(src["matrix"]["licensed"][0], query="Counterfactual", graph_class="Admg")
        )
        self.assertTrue(any("stays closed" in p for p in run(rows_of(src), src)))

    def test_the_committed_file_matches_the_sources(self) -> None:
        text = (Path(REPO_ROOT) / gen.OUTPUT).read_text(encoding="utf-8")
        self.assertEqual(text, gen.render(gen.build(gen.load_sources())))


if __name__ == "__main__":
    unittest.main()
