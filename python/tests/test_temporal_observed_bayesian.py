"""Observed-data posterior and simultaneous composite artifact axes."""

import json
from pathlib import Path

import antecedent
import numpy as np
import pytest
from antecedent.estimation import PreparedAnalysis
from antecedent.observation import IndependentGiven, RightCensored


def test_observed_temporal_posterior_and_response_roundtrip_together():
    pin = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "conformance/bayesian/temporal_observed_response/expected.json"
        ).read_text()
    )
    rng = np.random.default_rng(pin["seed"])
    x = rng.normal(size=pin["rows"])
    latent = (
        pin["intercept"]
        + pin["lag1_coefficient"] * np.roll(x, 1)
        + pin["lag2_coefficient"] * np.roll(x, 2)
        + pin["innovation_sd"] * rng.normal(size=x.size)
    )
    bound = 2.5 + rng.normal(size=x.size)
    data = {
        "x": x,
        "y": np.minimum(latent, bound),
        "c": bound,
        "r": (latent <= bound).astype(float),
    }
    query = antecedent.ResponseCurve(
        treatment="x",
        outcome="y",
        grid=pin["doses"],
        horizons=pin["horizons"],
        policy="pulse",
        treatment_lag=1,
        observation=RightCensored("y", "y", "c", "r"),
        observation_assumptions=[IndependentGiven([])],
    )
    plan = PreparedAnalysis.prepare(
        data,
        graph=[("x", 1, "y", 0), ("x", 2, "y", 0)],
        query=query,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=pin["draws"]),
        refute=False,
        bootstrap=0,
        seed=pin["posterior_seed"],
    )
    result = plan.estimate(data, seed=pin["posterior_seed"])
    assert result.response is not None
    # The response projection exposes bands; the composite artifact retains draws.
    decoded = antecedent.artifacts.loads(plan.export_artifact())
    assert decoded.payload_kind == "analysis_result"
    assert decoded.payload["estimate"] is None
    assert decoded.payload["response"] is not None
    assert decoded.payload["posterior_artifact"] is not None
    certificates = decoded.payload["temporal_identification"]
    assert [entry["horizon"] for entry in certificates] == pin["horizons"]
    namespace = decoded.payload["identification_variables"]
    assert len(namespace) > 4
    assert all(0 <= node["variable"] < 4 for node in namespace)
    assert any(node["offset"] < 0 for node in namespace)
    assert len({(node["variable"], node["offset"]) for node in namespace}) == len(namespace)

    again = antecedent.artifacts.loads(
        antecedent.artifacts.dumps(
            "analysis_result",
            decoded.payload,
            variable_names=decoded.variable_names,
            artifact_id="observed-response-roundtrip",
        )
    )
    assert again.payload == decoded.payload
    assert again.variable_names == decoded.variable_names
    values = np.asarray(result.response.values).reshape(-1)
    truth = [2 + 0.75 * x.mean(), 2 + 1.5 * x.mean(), 3.5 + 0.75 * x.mean(), 2.75 + 1.5 * x.mean()]
    assert values == pytest.approx(truth, abs=pin["mean_atol"])
