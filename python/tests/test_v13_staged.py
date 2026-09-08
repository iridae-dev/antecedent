"""Numerical and artifact evidence for the 1.3 staged cells."""

import json
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent import artifacts
from antecedent.errors import CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis
from antecedent.identify import identify

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    i = np.arange(500, dtype=float)
    a = np.sin(0.71 * i)
    m = 2 * a + np.cos(1.13 * i)
    y = 3 * a + 4 * m + 0.1 * np.sin(0.31 * i)
    data = {"a": a, "m": m, "y": y}
    return data, ac.Dag.from_edges(list(data), [("a", "m"), ("a", "y"), ("m", "y")])


@pytest.mark.parametrize(
    "contrast,key",
    [("natural_direct", "direct"), ("natural_indirect", "indirect"), ("total", "total")],
)
@pytest.mark.parametrize("accepted", [False, True])
def test_mediation_stages_and_artifacts(contrast, key, accepted):
    pin = json.loads((ROOT / "conformance/estimate/staged_static_kinds/expected.json").read_text())
    data, dag = fixture()
    graph = ac.AcceptedGraph(dag) if accepted else dag
    q = ac.MediationEffect(
        "a",
        "y",
        mediators=["m"],
        contrast=contrast,
        control_level=pin["control"],
        active_level=pin["active"],
    )
    identified = identify(graph=graph, query=q)
    assert identified
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=q, refute="full", bootstrap=0)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    result = prepared.estimate(data)
    assert result.effect == pytest.approx(pin[key], abs=pin["tolerance"])
    fresh = ac.analyze(data, graph=graph, query=q, refute="full", bootstrap=0)
    assert fresh.effect == pytest.approx(result.effect)
    assert result.validation.count == 3
    assert any("reused" in d for d in result.diagnostics)
    wire = artifacts.loads(prepared.export_artifact())
    assert wire.payload_kind == "static_result"
    assert wire.payload["estimate"] == result.effect
    assert wire.payload["assumptions"]
    assert len(wire.payload["refutations"]) == 3
    restored = artifacts.loads(
        artifacts.dumps(
            wire.payload_kind,
            wire.payload,
            variable_names=wire.variable_names,
            artifact_id="roundtrip",
        )
    )
    assert restored.payload == wire.payload


def test_counterfactual_stages_and_artifact():
    pin = json.loads((ROOT / "conformance/estimate/staged_static_kinds/expected.json").read_text())
    data, dag = fixture()
    q = ac.Counterfactual("a", "y", control_level=pin["control"], active_level=pin["active"])
    assert identify(graph=dag, query=q)
    prepared = PreparedAnalysis.prepare(data, graph=dag, query=q, refute="none")
    result = prepared.estimate(data)
    assert result.effect == pytest.approx(pin["counterfactual_mean"], abs=pin["tolerance"])
    assert len(result.unit_effects) == len(data["a"])
    assert ac.analyze(data, query=q, graph=dag, refute="none").effect == pytest.approx(
        result.effect
    )
    wire = artifacts.loads(prepared.export_artifact())
    assert wire.payload["control_level"] == pin["control"]
    assert wire.payload["standard_error"] is None
    assert wire.payload["unit_effects"] == result.unit_effects
    assert wire.payload["support"]
    with pytest.raises(CausalUnsupportedError):
        PreparedAnalysis.prepare(data, graph=ac.AcceptedGraph(dag), query=q)


@pytest.mark.parametrize(
    "kind", ["point", "elasticity", "semi", "average", "jacobian", "directional"]
)
def test_derivative_stages_and_artifacts(kind):
    pin = json.loads((ROOT / "conformance/response/staged_derivatives/expected.json").read_text())
    i = np.arange(800, dtype=float)
    a = 2 + np.sin(i * 0.71) + 0.2 * np.cos(i * 0.13)
    b = np.cos(i * 1.13)
    data = {"a": a, "b": b, "y": 5 + 2 * a - 0.5 * b, "v": 1 + 0.25 * a + 1.5 * b}
    graph = ac.Dag.from_edges(list(data), [("a", "y"), ("b", "y"), ("a", "v"), ("b", "v")])
    query = {
        "point": ac.PointDerivative("a", "y", at=2),
        "elasticity": ac.Elasticity("a", "y", at=2),
        "semi": ac.SemiElasticity("a", "y", at=2),
        "average": ac.AverageDerivative("a", "y"),
        "jacobian": ac.ResponseJacobian(["a", "b"], ["y", "v"], at=[2, 0]),
        "directional": ac.DirectionalDerivative(
            ["a", "b"], ["y", "v"], at=[2, 0], direction=[1, 2]
        ),
    }[kind]
    assert identify(graph=graph, query=query)
    config = {"bandwidth": 0.35} if kind in {"point", "elasticity", "semi"} else None
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=query, estimator_config=config)
    result = prepared.estimate(data)
    wire = artifacts.loads(prepared.export_artifact())
    assert wire.payload_kind == "response_result"
    assert result.identification
    # Native staged test pins all numeric coordinates; here both public routes must agree.
    fresh = ac.analyze(data, graph=graph, query=query, estimator_config=config, refute="none")
    assert fresh.estimate == result.estimate
    assert pin["point"] == 2
    with pytest.raises(CausalUnsupportedError):
        PreparedAnalysis.prepare(
            data, graph=graph, query=query, estimator_config=config, refute="full"
        )


@pytest.mark.parametrize("kind", ["selected", "right_km", "left_km", "right_cox", "left_cox"])
@pytest.mark.parametrize("accepted", [False, True])
def test_observation_pair_fixture(kind, accepted):
    from antecedent import observation as obs

    pin = json.loads((ROOT / f"conformance/response/observation_pairs/{kind}.json").read_text())
    rng = np.random.default_rng(pin["seed"])
    a = rng.uniform(-1, 1, pin["rows"])
    z = rng.uniform(-1, 1, pin["rows"])
    latent = 5 + 2 * a + z + rng.uniform(-0.5, 0.5, pin["rows"])
    if kind == "selected":
        event = (rng.random(pin["rows"]) < 1 / (1 + np.exp(-(0.4 * a + 0.3 * z)))).astype(float)
        data = {"a": a, "z": z, "y": np.where(event == 1, latent, np.nan), "event": event}
        mechanism = obs.Selected("y", "y", "event")
        assumption = obs.OutcomeIndependentGiven(["a", "z"])
        sign = 1
    else:
        rate = 0.07 * np.exp(0.6 * a + 0.5 * z) if kind.endswith("cox") else 0.07
        censoring = rng.exponential(1 / rate, pin["rows"])
        event = (latent <= censoring).astype(float)
        sign = -1 if kind.startswith("left") else 1
        data = {
            "a": a,
            "z": z,
            "y": sign * np.minimum(latent, censoring),
            "c": sign * censoring,
            "event": event,
        }
        cls = obs.LeftCensored if kind.startswith("left") else obs.RightCensored
        mechanism = cls("y", "y", "c", "event")
        assumption = obs.IndependentGiven(["a", "z"] if kind.endswith("cox") else [])
    query = ac.ResponseCurve(
        "a", "y", grid=pin["grid"], observation=mechanism, observation_assumptions=[assumption]
    )
    graph = ac.Dag.from_edges(list(data), [("a", "y"), ("z", "y")])
    graph = ac.AcceptedGraph(graph) if accepted else graph
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=query)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    result = prepared.estimate(data)
    fresh = ac.analyze(data, graph=graph, query=query, refute="none")
    assert fresh.response.values == result.response.values
    assert np.asarray(result.response.values).ravel() == pytest.approx(
        np.array(pin["mean"]) * sign, abs=pin["atol"]
    )
    assert result.uncertainty.kind == "none"
    wire = artifacts.loads(prepared.export_artifact())
    assert wire.payload_kind == "response_result"
    assert wire.payload["assumptions"]


def test_static_mediation_refuter_pin():
    from dataclasses import asdict

    pin = json.loads((ROOT / "conformance/estimate/staged_static_kinds/refuters.json").read_text())
    data, graph = fixture()
    query = ac.MediationEffect(
        "a", "y", mediators=["m"], contrast="natural_indirect", control_level=0.2, active_level=0.8
    )
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=query, refute="full", bootstrap=0)
    result = prepared.estimate(data, seed=pin["seed"])
    for report, expected in zip(result.validation.reports, pin["reports"], strict=True):
        actual = asdict(report)
        for key in ("original_ate", "refuted_ate", "comparison"):
            assert actual.pop(key) == pytest.approx(expected[key], abs=pin["atol"])
        assert actual == {k: v for k, v in expected.items() if k in actual}


def test_new_derivative_boundaries():
    with pytest.raises(ValueError):
        ac.Elasticity("a", "y", at=-0.5)
    data, graph = fixture()
    for query, config in [
        (ac.PointDerivative("a", "y", at=0.5), None),
        (ac.PointDerivative("a", "y", at=0.5), {"bandwidth": -0.2}),
        (ac.AverageDerivative("a", "y", weighting="custom"), None),
        (ac.ResponseJacobian(["a", "m", "y"], ["y"], at=[0, 0, 0]), None),
    ]:
        with pytest.raises((ValueError, ac.CausalError)):
            prepared = PreparedAnalysis.prepare(
                data, graph=graph, query=query, estimator_config=config
            )
            prepared.estimate(data)


def test_static_artifact_rejects_wrong_family_payload():
    data, graph = fixture()
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=ac.Counterfactual("a", "y"))
    prepared.estimate(data)
    wire = artifacts.loads(prepared.export_artifact())
    payload = dict(wire.payload)
    payload["unit_effects"] = None
    with pytest.raises((ValueError, ac.CausalError)):
        artifacts.dumps(
            "static_result", payload, variable_names=wire.variable_names, artifact_id="bad"
        )


@pytest.mark.parametrize("left", [False, True])
def test_python_consumes_cox_oracle(left):
    from antecedent import observation as obs

    pin = json.loads((ROOT / "conformance/response/conditional_ipcw/expected.json").read_text())
    rows = pin["data"]
    sign = -1 if left else 1
    observed = np.array(rows["time"])
    event = np.array(rows["event"])
    data = {
        "a": rows["a"],
        "z": rows["z"],
        "y": sign * observed,
        "c": sign * (observed + event),
        "event": event,
    }
    cls = obs.LeftCensored if left else obs.RightCensored
    query = ac.ResponseCurve(
        "a",
        "y",
        grid=[-0.5, 0.5],
        observation=cls("y", "y", "c", "event"),
        observation_assumptions=[obs.IndependentGiven(["a", "z"])],
    )
    adjusted = obs.adjusted_outcome(data, query, censoring_survival_floor=1e-6)
    assert adjusted.weights == pytest.approx(rows["weight"], abs=pin["atol"])


def test_counterfactual_prepared_refits_supplied_data():
    data, graph = fixture()
    query = ac.Counterfactual("a", "y", control_level=0.2, active_level=0.8)
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=query)
    original = prepared.estimate(data)
    changed = {**data, "y": 2 * data["y"]}
    refitted = prepared.estimate(changed)
    fresh = ac.analyze(changed, graph=graph, query=query, refute="none")
    assert refitted.effect == pytest.approx(2 * original.effect)
    assert refitted.unit_effects == pytest.approx(fresh.unit_effects)
    assert refitted.identification and refitted.assumptions and refitted.support
    assert np.isnan(refitted.estimate.se_analytic)
    assert refitted.estimate.se_bootstrap is None
    assert any("reused" in d for d in refitted.diagnostics)


def test_static_mediation_bootstrap_is_labelled_as_sampling_uncertainty():
    data, graph = fixture()
    query = ac.MediationEffect("a", "y", mediators=["m"], contrast="natural_direct")
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=query, bootstrap=20)
    result = prepared.estimate(data)
    assert np.isnan(result.estimate.se_analytic)
    assert result.estimate.se_bootstrap > 0
    wire = artifacts.loads(prepared.export_artifact())
    assert wire.payload["standard_error"] == result.estimate.se_bootstrap


def test_conditional_censoring_failures():
    from antecedent import observation as obs

    data = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "c": [1.0, 2.0, 3.0, 4.0],
        "d": [0.0, 1.0, 1.0, 1.0],
    }
    query = ac.ResponseCurve(
        "a",
        "y",
        grid=[1, 2],
        observation=obs.RightCensored("y", "y", "c", "d"),
        observation_assumptions=[obs.IndependentGiven(["a"])],
    )
    with pytest.raises(ac.CausalError, match="Cox"):
        obs.adjusted_outcome(data, query)
    with pytest.raises(ac.CausalError, match="delayed entry"):
        obs.adjusted_outcome(data, query, delayed_entry="a")


def test_static_mediation_recanting_witness_refuses():
    i = np.arange(100, dtype=float)
    data = {"a": np.sin(i), "l": np.cos(i), "m": np.sin(0.3 * i), "y": np.cos(0.2 * i)}
    graph = ac.Dag.from_edges(list(data), [("a", "l"), ("l", "m"), ("l", "y"), ("m", "y")])
    query = ac.MediationEffect("a", "y", mediators=["m"], contrast="natural_indirect")
    with pytest.raises(ac.CausalError):
        PreparedAnalysis.prepare(data, graph=graph, query=query)
