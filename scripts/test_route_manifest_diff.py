"""Route manifest diff against the published predecessor: tests on synthetic registries.

The tests are module-level functions so a scenario cell can cite one by name; `load_tests` hands
them to the unittest discovery that scripts/run_script_tests.py uses.
"""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import route_manifest_diff as rmd  # noqa: E402

AXES = {
    "queries": ["AverageEffect", "PulseEffect"],
    "graph_classes": ["Dag", "TemporalDag"],
    "structures": ["explicit"],
    "inferences": ["Frequentist"],
    "validations": ["none"],
}


def cell(query: str, graph: str) -> dict:
    return {
        "query": query, "graph_class": graph, "structure": "explicit",
        "inference": "Frequentist", "validation": "none",
    }


def registries(licensed=(), closed=(), n_a=(), graphless=()) -> dict:
    return {
        "axes": AXES,
        "licensed": {"cell": [cell(*c) for c in licensed]},
        "closed": {"closed": list(closed)},
        "n_a": {"n_a": list(n_a)},
        "graphless": {"license": [{"family": f, "design": d, "method": m} for f, d, m in graphless]},
    }


def test_a_licensed_coordinate_that_stops_running_is_a_lost_capability():
    old = registries(licensed=[("AverageEffect", "Dag")])
    new = registries()
    result = rmd.diff(old, new)
    assert [e["coordinate"][:2] for e in result["lost"]] == [("AverageEffect", "Dag")]
    assert result["lost"][0]["before"] == "licensed" and result["lost"][0]["after"] == "refused"


def test_a_licensed_coordinate_that_now_executes_unlicensed_is_not_lost():
    old = registries(licensed=[("AverageEffect", "Dag")])
    new = registries(
        closed=[{"queries": ["AverageEffect"], "graph_classes": ["Dag"], "executes_unlicensed": True,
                 "reason": "r"}]
    )
    assert rmd.diff(old, new)["lost"] == []


def test_a_refusal_that_becomes_a_capability_is_gained_and_a_class_change_is_listed():
    old = registries(n_a=[{"queries": ["PulseEffect"], "graph_classes": ["Dag"], "reason": "a"}])
    new = registries(
        licensed=[("AverageEffect", "Dag")],
        closed=[{"queries": ["PulseEffect"], "graph_classes": ["Dag"], "reason": "b"}],
    )
    result = rmd.diff(old, new)
    assert result["lost"] == []
    assert [(e["before"], e["after"]) for e in result["changed_refusals"]] == [("n_a", "refused")]
    assert [e["coordinate"][:2] for e in result["gained"]] == [("AverageEffect", "Dag")]


def test_a_dropped_graphless_licence_is_a_lost_capability():
    old = registries(graphless=[("interference", "cluster", "m")])
    result = rmd.diff(old, registries())
    assert [e["before"] for e in result["lost"]] == ["graphless_license"]


def test_the_working_tree_loses_no_capability_the_published_predecessor_had():
    result = rmd.build()
    assert result["coordinates"] > 4000
    assert result["lost"] == [], result["lost"][:3]
    assert result["changed_refusals"] == [], result["changed_refusals"][:3]


_CASES = [
    test_a_licensed_coordinate_that_stops_running_is_a_lost_capability,
    test_a_licensed_coordinate_that_now_executes_unlicensed_is_not_lost,
    test_a_refusal_that_becomes_a_capability_is_gained_and_a_class_change_is_listed,
    test_a_dropped_graphless_licence_is_a_lost_capability,
    test_the_working_tree_loses_no_capability_the_published_predecessor_had,
]


def load_tests(loader, tests, pattern):  # noqa: ARG001
    suite = unittest.TestSuite()
    suite.addTests(unittest.FunctionTestCase(case) for case in _CASES)
    return suite


if __name__ == "__main__":
    unittest.main()
