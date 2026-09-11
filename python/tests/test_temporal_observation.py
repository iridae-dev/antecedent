"""Licensed observation pairs on temporal ResponseCurve / InterventionResponse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent import observation as obs
from antecedent.errors import CausalError
from antecedent.estimation import PreparedAnalysis
from antecedent.intervention import Set

_ROOT = Path(__file__).resolve().parents[2]
_PIN = json.loads(
    (_ROOT / "conformance" / "response" / "temporal_observation" / "expected.json").read_text()
)
_EDGES = [("t", 1, "y", 0), ("t", 2, "y", 0)]
_TRUTH = np.asarray(_PIN["surface"]["mean"], dtype=float)
_ATOL = float(_PIN["atol"])
_NAIVE = float(_PIN["naive_gap_min"])


def _dgp() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_PIN["seed"])
    n = int(_PIN["rows"])
    t = rng.uniform(-1, 1, n)
    latent = np.zeros(n)
    for i in range(n):
        t1 = t[i - 1] if i >= 1 else 0.0
        t2 = t[i - 2] if i >= 2 else 0.0
        latent[i] = 5 + 2 * t1 + 3 * t2 + rng.uniform(-0.5, 0.5)
    t1 = np.concatenate([[0.0], t[:-1]])
    selected = (rng.random(n) < 1 / (1 + np.exp(-0.4 * t1))).astype(float)
    c_ind = rng.exponential(1 / 0.07, n)
    c_cox = rng.exponential(1 / (0.07 * np.exp(0.6 * t1)), n)
    return {
        "t": t,
        "latent": latent,
        "selected": selected,
        "c_ind": c_ind,
        "c_cox": c_cox,
    }


def _means(result) -> np.ndarray:
    assert result.response is not None
    return np.asarray([row[0] for row in result.response.values], dtype=float)


@pytest.mark.parametrize("kind", ["selected", "right_km", "right_cox"])
def test_temporal_observation_pair_consumes_fixture(kind):
    dgp = _dgp()
    t, latent = dgp["t"], dgp["latent"]
    if kind == "selected":
        y = np.where(dgp["selected"] == 1, latent, 0.0)
        data = {"t": t, "y": y, "r": dgp["selected"]}
        mechanism = obs.Selected("y", "y", "r")
        assumption = obs.OutcomeIndependentGiven(["t"])
    else:
        c = dgp["c_cox"] if kind.endswith("cox") else dgp["c_ind"]
        y = np.minimum(latent, c)
        data = {"t": t, "y": y, "c": c, "event": (latent <= c).astype(float)}
        mechanism = obs.RightCensored("y", "y", "c", "event")
        assumption = obs.IndependentGiven(["t"] if kind.endswith("cox") else [])
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=_PIN["grid"],
        horizons=_PIN["horizons"],
        observation=mechanism,
        observation_assumptions=[assumption],
    )
    prepared = PreparedAnalysis.prepare(data, graph=_EDGES, query=query, refute=False)
    result = prepared.estimate(data)
    fresh = antecedent.analyze(data, graph=_EDGES, query=query, refute=False)
    np.testing.assert_allclose(_means(fresh), _means(result), atol=1e-10)
    np.testing.assert_allclose(_means(result), _TRUTH, atol=_ATOL)
    assert result.uncertainty.kind == "none"
    naive = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.ResponseCurve("t", "y", grid=_PIN["grid"], horizons=_PIN["horizons"]),
        refute=False,
    )
    naive_err = np.max(np.abs(_means(naive) - _TRUTH))
    corr_err = np.max(np.abs(_means(result) - _TRUTH))
    assert naive_err > corr_err + _NAIVE


def test_temporal_right_censor_intervention_path():
    dgp = _dgp()
    y = np.minimum(dgp["latent"], dgp["c_cox"])
    data = {
        "t": dgp["t"],
        "y": y,
        "c": dgp["c_cox"],
        "event": (dgp["latent"] <= dgp["c_cox"]).astype(float),
    }
    query = antecedent.InterventionResponse(
        "y",
        intervention=Set("t", 0.5),
        horizons=_PIN["horizons"],
        observation=obs.RightCensored("y", "y", "c", "event"),
        observation_assumptions=[obs.IndependentGiven(["t"])],
    )
    result = antecedent.analyze(data, graph=_EDGES, query=query, refute=False)
    np.testing.assert_allclose(_means(result), _PIN["intervention_set_0_5"], atol=_ATOL)


def test_unlicensed_temporal_observation_refuses():
    dgp = _dgp()
    data = {"t": dgp["t"], "y": dgp["latent"], "lo": dgp["latent"] - 1, "hi": dgp["latent"] + 1}
    query = antecedent.ResponseCurve(
        "t",
        "y",
        grid=_PIN["grid"],
        horizons=_PIN["horizons"],
        observation=obs.IntervalCensored("y", "lo", "hi"),
        observation_assumptions=[obs.IndependentGiven([])],
    )
    with pytest.raises(CausalError, match="temporal response observation pair is not licensed"):
        antecedent.analyze(data, graph=_EDGES, query=query, refute=False)
