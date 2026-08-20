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
    "PulseEffect": (
        {
            "temporal_effect": {
                "treatment": 0,
                "outcome": 1,
                "policy": {"pulse": {"at": 0}},
                "control": {"set": {"variable": 0, "value": {"float64": 0.0}}},
                "active": {"set": {"variable": 0, "value": {"float64": 1.0}}},
                "horizon_steps": 1,
                "max_history_lag": None,
                "target_population": "all_observed",
            }
        },
        ["t", "y"],
    ),
    "SustainedEffect": (
        {
            "temporal_effect": {
                "treatment": 0,
                "outcome": 1,
                # Single-step: multi-step Sustained is estimator-refused (ADR 0021).
                "policy": {"sustained": {"from": 0, "until": 0}},
                "control": {"set": {"variable": 0, "value": {"float64": 0.0}}},
                "active": {"set": {"variable": 0, "value": {"float64": 1.0}}},
                "horizon_steps": 1,
                "max_history_lag": None,
                "target_population": "all_observed",
            }
        },
        ["t", "y"],
    ),
}

# Format-0.4 temporal attachment on a response functional. The `temporal` field is
# what distinguishes a licensed temporal ResponseCurve / InterventionResponse cell
# from its static namesake, so it needs its own round trip: the entries above cover
# the static shape only (absent `temporal` == static response).
_TEMPORAL_RESPONSE_PAYLOADS: dict[str, dict[str, object]] = {
    "ResponseCurve": {
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
            "temporal": {
                "horizons": [1, 2],
                "policy": {"pulse": {"at": 0}},
                "max_history_lag": 2,
            },
        }
    },
    "InterventionResponse": {
        "response": {
            "functional": {
                "intervention_response": {
                    "outcome": 1,
                    "interventions": [
                        {"set": {"variable": 0, "value": {"float64": 1.0}}}
                    ],
                }
            },
            "target_population": "all_observed",
            "observation": "complete",
            "observation_assumptions": [],
            # `max_history_lag` omitted on purpose: it is an optional wire field
            # (skip_serializing_if = Option::is_none), so an absent cap must survive
            # the round trip as absent rather than as an explicit null.
            "temporal": {
                "horizons": [1, 2],
                "policy": {"pulse": {"at": 0}},
            },
        }
    },
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


@pytest.mark.parametrize("query", sorted(_TEMPORAL_RESPONSE_PAYLOADS))
def test_temporal_response_attachment_round_trips(query: str) -> None:
    """Format-0.4 `temporal` attachment survives encode/decode unchanged."""
    payload = _TEMPORAL_RESPONSE_PAYLOADS[query]
    encoded = artifacts.dumps(
        "query", payload, variable_names=["t", "y"], artifact_id=f"{query}_temporal"
    )
    decoded = artifacts.loads(encoded)
    assert decoded.payload == payload
    assert decoded.payload["response"]["temporal"]["horizons"] == [1, 2]
    assert decoded.payload["response"]["temporal"]["policy"] == {"pulse": {"at": 0}}


def test_response_without_temporal_field_decodes_as_static() -> None:
    """A format-0.3 response payload (no `temporal` key) stays static, not temporal."""
    static_payload, names = _QUERY_PAYLOADS["ResponseCurve"]
    decoded = artifacts.loads(
        artifacts.dumps("query", static_payload, variable_names=names, artifact_id="static")
    )
    assert "temporal" not in decoded.payload["response"]
    assert decoded.payload == static_payload
