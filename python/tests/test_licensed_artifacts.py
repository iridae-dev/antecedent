"""Artifact round trips for licensed matrix queries only."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from antecedent import artifacts

_ROOT = Path(__file__).resolve().parents[2]
_LICENSED = tomllib.loads((_ROOT / "parity" / "support_licensed.toml").read_text())

_QUERY_PAYLOADS: dict[str, tuple[dict[str, object], list[str]]] = {
    "AverageEffect": (
        {
            "average_effect": {
                "treatment": 0,
                "outcome": 1,
                "effect_modifiers": [],
                "control": {"set": {"variable": 0, "value": {"float64": 0.0}}},
                "active": {"set": {"variable": 0, "value": {"float64": 1.0}}},
                "target_population": "all_observed",
            }
        },
        ["t", "y"],
    ),
    "ResponseCurve": (
        {
            "response": {
                "functional": {
                    "mean_curve": {
                        "outcome": 1,
                        "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}},
                    }
                },
                "target_population": "all_observed",
                "observation": "complete",
                "observation_assumptions": [],
            }
        },
        ["t", "y"],
    ),
    "ConditionalEffect": (
        {
            "conditional_effect": {
                "inner": {
                    "average_effect": {
                        "treatment": 0,
                        "outcome": 1,
                        "effect_modifiers": [2],
                        "control": {"set": {"variable": 0, "value": {"float64": 0.0}}},
                        "active": {"set": {"variable": 0, "value": {"float64": 1.0}}},
                        "target_population": "all_observed",
                    }
                }
            }
        },
        ["t", "y", "w"],
    ),
    "PathSpecificEffect": (
        {
            "path_specific": {
                "treatment": 0,
                "outcome": 1,
                "path_nodes": [2],
                "control": {"set": {"variable": 0, "value": {"float64": 0.0}}},
                "active": {"set": {"variable": 0, "value": {"float64": 1.0}}},
                "target_population": "all_observed",
                "max_paths": 64,
                "max_len": 16,
            }
        },
        ["t", "y", "m"],
    ),
    "InterventionalDistribution": (
        {
            "distribution": {
                "outcomes": [1],
                "interventions": [{"set": {"variable": 0, "value": {"float64": 1.0}}}],
                "conditioning": [],
                "target_population": "all_observed",
            }
        },
        ["t", "y"],
    ),
    "InterventionResponse": (
        {
            "response": {
                "functional": {
                    "intervention_response": {
                        "outcome": 1,
                        "interventions": [{"set": {"variable": 0, "value": {"float64": 0.25}}}],
                    }
                },
                "target_population": "all_observed",
                "observation": "complete",
                "observation_assumptions": [],
            }
        },
        ["t", "y"],
    ),
}


def _licensed_queries() -> list[str]:
    return sorted({row["query"] for row in _LICENSED.get("cell") or []})


@pytest.mark.parametrize("query", _licensed_queries())
def test_licensed_query_artifact_round_trips(query: str) -> None:
    assert query in _QUERY_PAYLOADS, f"licensed query {query} has no artifact payload"
    payload, names = _QUERY_PAYLOADS[query]
    encoded = artifacts.dumps("query", payload, variable_names=names, artifact_id=query)
    decoded = artifacts.loads(encoded)
    assert decoded.payload == payload
    again = artifacts.loads(
        artifacts.dumps(
            decoded.payload_kind,
            decoded.payload,
            variable_names=decoded.variable_names,
            artifact_id=decoded.artifact_id,
        )
    )
    assert again == decoded
