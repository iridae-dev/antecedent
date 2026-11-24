"""Source-level integration assertions for the compiler-owned events workflow."""

import math

import pytest
from antecedent import events


def request() -> dict:
    return {
        "operation": "forecast",
        "request": {
            "history": {
                "catalog": ["mention"],
                "time_unit": "hour",
                "origin": "launch",
                "source": "news",
                "empty_origin": 0.0,
                "window": {"start": 0.0, "end": 1.0},
                "left": [],
                "events": [],
            },
            "forecast": {"start": 1.0, "end": 3.0},
            "source": {"kind": "fixture", "paths": [[{"time": 1.25, "kind": 0}], []]},
            "paths": 2,
            "max_events": 8,
            "seed": 33,
        },
    }


def test_event_forecast_replays_and_rejects_changed_path():
    declaration = request()
    result = events.prepare(declaration).execute()
    identity = result.report()["program"]
    assert events.replay(declaration, identity).paths() == result.paths()
    declaration["request"]["source"]["paths"][0][0]["time"] = 1.5
    with pytest.raises(ValueError, match="replay program identity changed"):
        events.replay(declaration, identity)


def test_off_grid_event_path_composes_with_existing_distribution():
    law = events.prepare(request()).execute()
    worlds = {
        "system": {
            "variables": ["load"],
            "inputs": [
                {
                    "name": "pressure",
                    "interpolation": "piecewise_constant",
                    "times": [1.0],
                    "values": [0.0],
                    "envelope": [0.0, 10.0],
                }
            ],
            "drift": [{"op": "variable", "variable": 1}],
            "diffusion": [{"expression": {"op": "constant", "value": 0.0}, "square_root": False}],
        },
        "population": "launch",
        "horizon": 2,
        "time_unit": "hour",
        "origin": "launch",
        "policy": {
            "start": 1.0,
            "step": 1.0,
            "steps": 2,
            "scheme": "euler_maruyama",
            "paths": 2,
            "seed": 8,
        },
        "initial": [0.0],
        "worlds": [],
        "bindings": [{"kind": 0, "input": "pressure", "magnitude": 1.0, "decay": 1.0}],
        "outputs": [
            {
                "state": "load",
                "world": "baseline",
                "variable_id": "launch.load",
                "units": "score",
                "functional": {"kind": "maximum"},
            }
        ],
    }
    result = events.prepare({"operation": "worlds", "worlds": worlds}, source=law).execute()
    assert isinstance(result.distribution(), bytes)
    assert result.report()["event_meaning"] == "fixture"
    expected_coarse = math.exp(-0.75)
    expected_refined = 0.5 * (math.exp(-0.25) + math.exp(-0.75) + math.exp(-1.25))
    assert result.report()["grid"][0]["mean"] == pytest.approx(
        (expected_coarse - expected_refined) / 2.0
    )


def test_horizon_offspring_is_not_the_lifetime_mass_or_the_discounted_resolvent():
    horizon = 2.0
    alpha = 0.4
    report = events.prepare(
        {
            "operation": "offspring",
            "parameters": {"baseline": [0.5], "branching": [alpha], "decay": [1.0]},
            "source": 0,
            "horizon": horizon,
        }
    ).execute().report()
    direct = alpha * (1.0 - math.exp(-horizon))
    lifetime = alpha / (1.0 - alpha)
    exact = lifetime * (1.0 - math.exp(-(1.0 - alpha) * horizon))
    discounted = direct / (1.0 - direct)
    assert report["direct"][0] == pytest.approx(direct)
    assert report["lifetime_descendants"][0] == pytest.approx(lifetime)
    assert report["horizon_descendants"][0] == pytest.approx(exact)
    assert abs(report["horizon_descendants"][0] - discounted) > 1e-3


def test_excitation_and_marked_plans_run_under_their_own_units():
    offspring = events.prepare(
        {
            "operation": "offspring",
            "parameters": {"baseline": [0.5], "branching": [0.4], "decay": [1.0]},
            "source": 0,
            "horizon": 1.0,
        }
    )
    assert offspring.unit_support()["unit"] == "forecast.hawkes_excitation"
    assert offspring.inspect()["route"] == "HawkesExcitation"
    marked = events.prepare(
        {
            "operation": "marked_likelihood",
            "history": {
                "time_unit": "hour",
                "origin": "origin",
                "source": "log",
                "empty_origin": 0.0,
                "window": {"start": 0.0, "end": 1.0},
                "left": [],
                "events": [{"time": 0.5, "prototype": 0}],
                "prototypes": [{"source": [1.0], "target": [1.0], "mass": 1.0}],
            },
            "parameters": {"baseline": [1.0], "excitation": [0.2], "decay": [1.0]},
        }
    )
    assert marked.unit_support()["unit"] == "forecast.marked_event_process"


def _pair_history() -> dict:
    return {
        "catalog": ["release", "activity"],
        "time_unit": "hour",
        "origin": "launch",
        "source": "log",
        "empty_origin": 0.0,
        "window": {"start": 0.0, "end": 1.0},
        "left": [],
        "events": [
            {"time": 0.10, "kind": 0},
            {"time": 0.16, "kind": 1},
            {"time": 0.22, "kind": 1},
            {"time": 0.50, "kind": 0},
            {"time": 0.56, "kind": 1},
            {"time": 0.62, "kind": 1},
            {"time": 0.80, "kind": 0},
            {"time": 0.86, "kind": 1},
            {"time": 0.92, "kind": 1},
        ],
    }


def _provenance() -> dict[str, str]:
    digest = "01" * 32
    return {
        "method": "pc",
        "inference": "frequentist",
        "discovered_class": "dag",
        "query": digest,
        "search_bounds": "alpha=0.05;max_cond_size=1;max_lag=1;schedule=none",
        "data_snapshot": "02" * 32,
        "program": "03" * 32,
        "claim": "04" * 32,
        "review": "accepted",
        "resolved_marks": "0",
    }


def test_excluded_shock_returns_a_lag_query_without_an_excitation_coefficient():
    report = events.prepare(
        {
            "operation": "shock",
            "history": _pair_history(),
            "feature": {"knots": [0.0, 0.4], "values": [0.0, 1.0]},
            "source": 0,
            "target": 1,
            "horizon": 1.0,
            "instrument": "launch_window",
        }
    ).execute().report()
    assert report == {
        "exposure": "release",
        "outcome": "activity",
        "horizon": 1.0,
        "instrument": "launch_window",
    }


def test_excitation_question_uses_the_discovery_provenance():
    from antecedent import Dag

    question = {
        "operation": "excitation_question",
        "challenge": {
            "history": _pair_history(),
            "fit": {
                "decay": [0.4, 1.7, 1.0, 0.3],
                "column_cap": 0.9,
                "iterations": 800,
            },
            "alternative": {"knots": [0.0, 0.4], "values": [0.0, 1.0]},
            "source": 0,
            "target": 1,
            "horizon": 1.0,
            "tolerance": 0.05,
        },
        "bridge": {
            "source": {
                "variable_id": "edge.release_to_activity",
                "variable_name": "direct offspring",
                "units": "count",
                "population_id": "observed_stream",
                "regime_id": "observational",
                "horizon": 1,
                "functional_id": "direct_offspring",
            },
            "target": {
                "variable_id": "edge.release_to_activity",
                "variable_name": "direct offspring",
                "units": "count",
                "population_id": "observed_stream",
                "regime_id": "do_one_release",
                "horizon": 1,
                "functional_id": "direct_offspring",
            },
            "slope": 2.0,
            "intercept": 0.0,
            "statement": "one added event yields this direct mass under the declared kernel",
        },
        "action": {"kind": "now"},
        "draw": {
            "release": 0.0,
            "window_end": 2.0,
            "paths": 1,
            "max_events": 4,
            "seed": 7,
        },
    }
    with pytest.raises(ValueError, match="discovered graph"):
        events.prepare(question)
    graph = Dag.from_edges(["release", "activity"], [("release", "activity")])
    plan = events.prepare(question, graph=graph, provenance=_provenance())
    evidence = plan.inspect()["evidence"]
    assert isinstance(evidence, str) and len(evidence) == 64
    report = plan.execute().report()
    assert report["accepted"] is False
    assert report["reading"]["status"] == "withheld"
    assert report["reading"]["reason"] == "immigrant-alternative fit is weakly identified"
    assert "bridged_direct" not in report
