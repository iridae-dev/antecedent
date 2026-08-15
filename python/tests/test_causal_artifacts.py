"""Cross-language conformance for format-0.3 causal payload artifacts."""

from __future__ import annotations

import json

import pytest
from antecedent import artifacts
from antecedent._native import decode_causal_artifact, encode_causal_artifact


def _response_query(functional: object) -> dict[str, object]:
    return {
        "response": {
            "functional": functional,
            "target_population": "all_observed",
            "observation": "complete",
            "observation_assumptions": [],
        }
    }


@pytest.mark.parametrize(
    "functional",
    [
        {
            "mean_curve": {
                "outcome": 2,
                "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}},
            }
        },
        {"average_derivative": {"outcome": 2, "treatment": 0, "weighting": "observed"}},
        {
            "point_derivative": {
                "outcome": 2,
                "treatment": 0,
                "at": 0.5,
                "order": 1,
                "scale": "identity",
            }
        },
        {
            "directional_derivative": {
                "outcomes": [2],
                "treatments": [0, 1],
                "at": [0.0, 1.0],
                "direction": [1.0, -1.0],
            }
        },
        {
            "jacobian": {
                "outcomes": [2],
                "treatments": [0, 1],
                "at": [0.5, 1.0],
                "scale": "log_treatment",
            }
        },
        {
            "intervention_response": {
                "outcome": 2,
                "interventions": [{"set": {"variable": 0, "value": {"float64": 1.0}}}],
            }
        },
    ],
)
def test_every_response_functional_crosses_rust_and_python(functional: object) -> None:
    payload = _response_query(functional)
    encoded = artifacts.dumps(
        "query", payload, variable_names=["a", "b", "y"], artifact_id="response-query"
    )
    decoded = artifacts.loads(encoded)
    assert decoded.format_version == (0, 3)
    assert decoded.payload == payload
    assert (
        artifacts.loads(
            artifacts.dumps(
                decoded.payload_kind,
                decoded.payload,
                variable_names=decoded.variable_names,
                artifact_id=decoded.artifact_id,
            )
        )
        == decoded
    )


@pytest.mark.parametrize(
    ("observation", "assumptions"),
    [
        ("complete", []),
        (
            {"right_censored": {"latent": 2, "observed": 3, "censoring": 4, "event": 5}},
            [{"independent_given": []}],
        ),
        (
            {"left_censored": {"latent": 2, "observed": 3, "censoring": 4, "event": 5}},
            [{"independent_given": [1]}],
        ),
        (
            {"interval_censored": {"latent": 2, "lower": 3, "upper": 4}},
            [{"structural": "gaussian_interval"}],
        ),
        (
            {"truncated": {"latent": 2, "observed": 3, "lower": 4, "upper": None}},
            [{"structural": "gaussian_truncation"}],
        ),
        (
            {"selected": {"latent": 2, "observed": 3, "indicator": 4}},
            [{"outcome_independent_given": [1]}],
        ),
    ],
)
def test_every_observation_mechanism_and_assumption_variant(
    observation: object, assumptions: object
) -> None:
    payload = _response_query(
        {"mean_curve": {"outcome": 2, "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}}}}
    )
    response = payload["response"]
    assert isinstance(response, dict)
    response["observation"] = observation
    response["observation_assumptions"] = assumptions
    decoded = artifacts.loads(
        artifacts.dumps(
            "query",
            payload,
            variable_names=["a", "x", "y", "observed", "censor", "event"],
            artifact_id="observation-query",
        )
    )
    assert decoded.payload == payload


@pytest.mark.parametrize(
    "functional",
    [
        {
            "mean_curve": {
                "outcome": 1,
                "treatment": {
                    "variable": 0,
                    "grid": {"linspace": {"start": 0.0, "end": 1.0, "points": 3}},
                },
            }
        },
        {
            "average_derivative": {
                "outcome": 1,
                "treatment": 0,
                "weighting": {"uniform": {"lower": 0.0, "upper": 1.0}},
            }
        },
        {
            "average_derivative": {
                "outcome": 1,
                "treatment": 0,
                "weighting": {"custom": [0.25, 0.75]},
            }
        },
        {
            "point_derivative": {
                "outcome": 1,
                "treatment": 0,
                "at": 1.0,
                "order": 1,
                "scale": "log_treatment",
            }
        },
        {
            "point_derivative": {
                "outcome": 1,
                "treatment": 0,
                "at": 1.0,
                "order": 1,
                "scale": "log_outcome",
            }
        },
        {
            "point_derivative": {
                "outcome": 1,
                "treatment": 0,
                "at": 1.0,
                "order": 1,
                "scale": "log_log",
            }
        },
    ],
)
def test_every_grid_weighting_and_derivative_scale_wire_variant(functional: object) -> None:
    payload = _response_query(functional)
    assert (
        artifacts.loads(
            artifacts.dumps(
                "query", payload, variable_names=["a", "y"], artifact_id="response-options"
            )
        ).payload
        == payload
    )


def test_native_encode_causal_artifact_returns_real_bytes() -> None:
    """`encode_causal_artifact` must return Python `bytes`, not `list[int]`.

    A bare PyO3 `Vec<u8>` return converts to `list[int]`, which is not what
    the `bytes` return-type annotation promises and forced `artifacts.dumps`
    to defensively wrap every call in `bytes(...)`. The Rust boundary now
    returns real `PyBytes` directly.
    """
    payload = _response_query(
        {"mean_curve": {"outcome": 1, "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}}}}
    )
    encoded = encode_causal_artifact(
        "query",
        ["a", "y"],
        json.dumps(payload, allow_nan=False, separators=(",", ":")),
        "response-query",
    )
    assert isinstance(encoded, bytes)
    assert not isinstance(encoded, list)

    decoded = decode_causal_artifact(encoded)
    assert decoded.payload_kind == "query"
    assert decoded.artifact_id == "response-query"


def test_artifacts_dumps_round_trips_through_public_api_and_returns_bytes() -> None:
    payload = _response_query(
        {"mean_curve": {"outcome": 1, "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}}}}
    )
    encoded = artifacts.dumps(
        "query", payload, variable_names=["a", "y"], artifact_id="response-query"
    )
    assert isinstance(encoded, bytes)

    decoded = artifacts.loads(encoded)
    assert decoded.payload == payload
    assert decoded.artifact_id == "response-query"


def test_transport_query_crosses_rust_and_python() -> None:
    response = _response_query(
        {"mean_curve": {"outcome": 1, "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}}}}
    )["response"]
    payload = {
        "transport": {
            "response": response,
            "source_population": "trial",
            "target_population": "target",
            "source_experiments": [0],
        }
    }
    decoded = artifacts.loads(
        artifacts.dumps("query", payload, variable_names=["a", "y"], artifact_id="q")
    )
    assert decoded.payload == payload


@pytest.mark.parametrize(
    "assignment",
    [
        {"bernoulli": {"probabilities": [0.4, 0.6]}},
        {"complete_randomization": {"treated": 2}},
        {"cluster_randomization": {"clusters": [0, 0, 1, 1], "treated_clusters": 1}},
    ],
)
@pytest.mark.parametrize(
    "exposure",
    [
        "own_treatment",
        "neighbor_count",
        "neighbor_fraction",
        "weighted_neighbor_exposure",
        {"custom": "registry.mapping"},
    ],
)
def test_every_interference_assignment_and_exposure_query_variant(
    assignment: object, exposure: object
) -> None:
    payload = {
        "interference": {
            "assignment": assignment,
            "exposure": exposure,
            "functional": {
                "exposure_contrast": {
                    "outcome": 1,
                    "from": {"own": 0.0, "neighbors": 0.0},
                    "to": {"own": 0.0, "neighbors": 1.0},
                }
            },
            "probability_draws": 500,
        }
    }
    decoded = artifacts.loads(
        artifacts.dumps("query", payload, variable_names=["a", "y"], artifact_id="q")
    )
    assert decoded.payload == payload


def _response_result(
    *,
    estimand: object,
    status: str,
    estimate: object,
    uncertainty: object,
    support_dim: int = 1,
) -> dict[str, object]:
    return {
        "estimand": estimand,
        "identification_status": status,
        "estimate": estimate,
        "uncertainty": uncertainty,
        "support": {
            "status": "supported",
            "query_region": {
                "minima": [0.0] * support_dim,
                "maxima": [1.0] * support_dim,
            },
            "diagnostics": [],
            "warnings": [],
        },
        "assumptions": [],
        "provenance_id": "estimate.response.test",
    }


_MEAN_CURVE = {
    "mean_curve": {
        "outcome": 1,
        "treatment": {"variable": 0, "grid": {"values": [0.0, 1.0]}},
    }
}
_SURFACE = {"surface": {"grid": [0.0, 1.0], "dimension": 1, "mean": [1.0, 2.0]}}
_ENVELOPE = {
    "envelope": {
        "grid": [0.0, 1.0],
        "dimension": 1,
        "lower": [0.0, 1.0],
        "upper": [1.0, 2.0],
    }
}


@pytest.mark.parametrize(
    ("estimand", "status", "estimate", "uncertainty", "support_dim", "variable_names"),
    [
        (
            _MEAN_CURVE,
            "nonparametrically_identified",
            {"point_identified": _SURFACE},
            "none",
            1,
            ["a", "y"],
        ),
        (
            _MEAN_CURVE,
            "nonparametrically_identified",
            {"point_identified": _SURFACE},
            {"pointwise_band": {"level": 0.95, "lower": [0.0, 1.0], "upper": [1.0, 2.0]}},
            1,
            ["a", "y"],
        ),
        (
            _MEAN_CURVE,
            "nonparametrically_identified",
            {"point_identified": _SURFACE},
            {
                "simultaneous_band": {
                    "level": 0.95,
                    "lower": [0.0, 1.0],
                    "upper": [1.0, 2.0],
                    "replicates": 500,
                }
            },
            1,
            ["a", "y"],
        ),
        (
            _MEAN_CURVE,
            "nonparametrically_identified",
            {"point_identified": _SURFACE},
            {
                "identified_envelope_band": {
                    "level": 0.95,
                    "lower_outer": [0.0, 1.0],
                    "upper_outer": [1.0, 2.0],
                }
            },
            1,
            ["a", "y"],
        ),
        (
            _MEAN_CURVE,
            "nonparametrically_identified",
            {"point_identified": _SURFACE},
            {"posterior": {"artifact_id": "posterior-1"}},
            1,
            ["a", "y"],
        ),
        (
            _MEAN_CURVE,
            "partially_identified",
            {"partially_identified": _ENVELOPE},
            "none",
            1,
            ["a", "y"],
        ),
        (
            {"average_derivative": {"outcome": 1, "treatment": 0, "weighting": "observed"}},
            "nonparametrically_identified",
            {"point_identified": {"scalar": 1.0}},
            {
                "scalar": {
                    "standard_error": 0.1,
                    "level": 0.95,
                    "lower": 0.8,
                    "upper": 1.2,
                }
            },
            1,
            ["a", "y"],
        ),
        (
            {
                "directional_derivative": {
                    "outcomes": [1],
                    "treatments": [0],
                    "at": [0.5],
                    "direction": [1.0],
                }
            },
            "partially_identified",
            {"partially_identified": {"vector": [1.0]}},
            "none",
            1,
            ["a", "y"],
        ),
        (
            {
                "jacobian": {
                    "outcomes": [2],
                    "treatments": [0, 1],
                    "at": [0.5, 1.0],
                    "scale": "identity",
                }
            },
            "nonparametrically_identified",
            {
                "point_identified": {
                    "jacobian": {"outcomes": 1, "treatments": 2, "values": [1.0, 2.0]}
                }
            },
            "none",
            2,
            ["a", "b", "y"],
        ),
        (
            _MEAN_CURVE,
            "graph_dependent",
            {"graph_dependent": [[7, _SURFACE], [9, _SURFACE]]},
            "none",
            1,
            ["a", "y"],
        ),
        (
            _MEAN_CURVE,
            "not_identified",
            {"unidentified": {"certificate": "not identified"}},
            "none",
            1,
            ["a", "y"],
        ),
    ],
)
def test_every_response_result_wire_variant_round_trips(
    estimand: object,
    status: str,
    estimate: object,
    uncertainty: object,
    support_dim: int,
    variable_names: list[str],
) -> None:
    payload = _response_result(
        estimand=estimand,
        status=status,
        estimate=estimate,
        uncertainty=uncertainty,
        support_dim=support_dim,
    )
    decoded = artifacts.loads(
        artifacts.dumps(
            "response_result",
            payload,
            variable_names=variable_names,
            artifact_id="response",
        )
    )
    assert decoded.payload == payload


@pytest.mark.parametrize(
    ("kind", "payload", "variable_names"),
    [
        (
            "transport_identification",
            {
                "not_certified": {
                    "reason": "subset",
                    "witness": [0],
                    "message": "no impossibility claim",
                }
            },
            ["a", "y"],
        ),
        (
            "transport_identification",
            {
                "transportable": {
                    "formula": {
                        "standardize": {
                            "over": [2],
                            "source_response": {
                                "population": "trial",
                                "variables": [1],
                                "conditioned_on": [2],
                                "interventions": [0],
                            },
                            "target_law": {
                                "population": "target",
                                "variables": [2],
                                "conditioned_on": [],
                                "interventions": [],
                            },
                        }
                    },
                    "certificate": {
                        "rule": "standardize",
                        "selection_targets": [2],
                        "premises": [],
                    },
                }
            },
            ["a", "y", "z"],
        ),
        (
            "transport_identification",
            {
                "transportable": {
                    "formula": {
                        "recursive_factorization": {
                            "sum_out": [2],
                            "factors": [
                                {
                                    "population": "trial",
                                    "variables": [1],
                                    "conditioned_on": [2],
                                    "interventions": [0],
                                },
                                {
                                    "population": "target",
                                    "variables": [2],
                                    "conditioned_on": [],
                                    "interventions": [],
                                },
                            ],
                        }
                    },
                    "certificate": {
                        "rule": "recursive",
                        "selection_targets": [2],
                        "premises": [],
                    },
                }
            },
            ["a", "y", "z"],
        ),
        (
            "transport_identification",
            {
                "transportable": {
                    "formula": {
                        "direct": {
                            "population": "trial",
                            "variables": [1],
                            "conditioned_on": [],
                            "interventions": [0],
                        }
                    },
                    "certificate": {
                        "rule": "direct",
                        "selection_targets": [],
                        "premises": ["same mechanism"],
                    },
                }
            },
            ["a", "y"],
        ),
        (
            "transport_estimate",
            {
                "ipw": 1.1,
                "aipw": 1.0,
                "selection_overlap": {
                    "probability_min": 0.2,
                    "probability_max": 0.8,
                    "effective_sample_size": 20.0,
                    "extreme_weight_count": 1,
                },
                "treatment_overlap": {
                    "probability_min": 0.3,
                    "probability_max": 0.7,
                    "effective_sample_size": 24.0,
                    "extreme_weight_count": 0,
                },
            },
            ["a", "y"],
        ),
        (
            "interference_estimate",
            {
                "contrast": {
                    "horvitz_thompson": 1.0,
                    "hajek": 0.9,
                    "conservative_variance": 0.2,
                },
                "from_probability_method": "exact",
                "to_probability_method": {"monte_carlo": {"draws": 500, "seed": 7}},
                "minimum_exposure_probability": 0.1,
            },
            ["a", "y"],
        ),
    ],
)
def test_transport_and_interference_result_artifacts(
    kind: str, payload: dict[str, object], variable_names: list[str]
) -> None:
    decoded = artifacts.loads(
        artifacts.dumps(kind, payload, variable_names=variable_names, artifact_id="result")  # type: ignore[arg-type]
    )
    assert decoded.payload == payload
