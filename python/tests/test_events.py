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
