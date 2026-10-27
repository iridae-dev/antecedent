"""Numerical and artifact evidence for the staged cells."""

import antecedent as ac
import numpy as np
import pytest
from antecedent import artifacts
from antecedent.errors import CausalUnsupportedError
from antecedent.identify import identify
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal
from _repo_text import REPO_ROOT, load_json


def _prepare(data, *, bootstrap=None, **declarations):
    """Prepare with the replicate budget declared on the scientific inference."""
    if bootstrap is not None:
        declarations["inference"] = ac.Frequentist(bootstrap=bootstrap)
    return ac.prepare(data, **declarations)


def _estimate(data, **declarations):
    return ac.estimate(_prepare(data, **declarations))


ROOT = REPO_ROOT


def fixture():
    i = np.arange(500, dtype=float)
    a = np.sin(0.71 * i)
    m = 2 * a + np.cos(1.13 * i)
    y = 3 * a + 4 * m + 0.1 * np.sin(0.31 * i)
    data = {"a": a, "m": m, "y": y}
    return data, ac.Dag.from_edges(list(data), [("a", "m"), ("a", "y"), ("m", "y")])


def confounded_derivatives():
    i = np.arange(800, dtype=float)
    z = np.cos(i * 0.41)
    a = 2 + 0.6 * z + np.sin(i * 0.71) + 0.2 * np.cos(i * 0.13)
    data = {"a": a, "z": z, "y": 5 + 2 * a + 3 * z}
    graph = ac.Dag.from_edges(list(data), [("z", "a"), ("z", "y"), ("a", "y")])
    return data, graph


def confounded_static():
    pin = load_json(ROOT / "conformance/estimate/staged_static_kinds/confounded.json")
    i = np.arange(500, dtype=float)
    z = np.cos(i * 0.41)
    a = np.sin(i * 0.71) + 0.4 * z
    m = 2 * a + 0.5 * z + np.cos(i * 1.13)
    y = 3 * a + 4 * m + 5 * z + 0.1 * np.sin(i * 0.31)
    data = {"a": a, "m": m, "y": y, "z": z}
    graph = ac.Dag.from_edges(
        list(data), [("a", "m"), ("a", "y"), ("m", "y"), ("z", "a"), ("z", "m"), ("z", "y")]
    )
    return pin, data, graph


@pytest.mark.parametrize(
    "contrast,key",
    [("natural_direct", "direct"), ("natural_indirect", "indirect"), ("total", "total")],
)
@pytest.mark.parametrize("accepted", [False, True])
def test_mediation_stages_and_artifacts(contrast, key, accepted):
    pin = load_json(ROOT / "conformance/estimate/staged_static_kinds/expected.json")
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
    prepared = _prepare(data, graph=graph, query=q, refute="full", bootstrap=0)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    outcome = ac.estimate(prepared)
    result = require_success(outcome)
    assert result.effect == pytest.approx(pin[key], abs=pin["tolerance"])
    assert result.estimate.estimator_id == "mediation.linear"
    fresh = require_success(
        ac.analyze(data, graph=graph, query=q, refute="full", inference=ac.Frequentist(bootstrap=0))
    )
    assert fresh.effect == pytest.approx(result.effect)
    assert result.validation.count == 3
    assert any("reused" in d for d in result.diagnostics)
    wire = artifacts.loads(bytes(outcome.export(quantity="record")))
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
    pin = load_json(ROOT / "conformance/estimate/staged_static_kinds/expected.json")
    data, dag = fixture()
    q = ac.Counterfactual("a", "y", control_level=pin["control"], active_level=pin["active"])
    assert identify(graph=dag, query=q)
    prepared = _prepare(data, graph=dag, query=q, refute="none")
    outcome = ac.estimate(prepared)
    result = require_success(outcome)
    assert result.effect == pytest.approx(pin["counterfactual_mean"], abs=pin["tolerance"])
    assert result.estimate.estimator_id == "gcm.fit"
    assert len(result.unit_effects) == len(data["a"])
    text = repr(result)
    assert "mean_ite=" in text
    assert "±nan" not in text
    assert result.mean_ite == pytest.approx(result.effect)
    assert require_success(
        ac.analyze(data, query=q, graph=dag, refute="none")
    ).effect == pytest.approx(result.effect)
    wire = artifacts.loads(bytes(outcome.export(quantity="record")))
    assert wire.payload["control_level"] == pin["control"]
    assert wire.payload["standard_error"] is None
    assert tuple(wire.payload["unit_effects"]) == result.unit_effects
    assert wire.payload["support"]
    # An accepted DAG runs the same GCM path as the explicit one.
    accepted = require_success(_estimate(data, graph=ac.AcceptedGraph(dag), query=q))
    assert accepted.estimate.estimator_id == "gcm.fit"
    assert accepted.effect == pytest.approx(result.effect, abs=1e-12)
    assert accepted.unit_effects == pytest.approx(result.unit_effects, abs=1e-12)


def test_nested_counterfactual_stages_shared_draws_artifact_and_refresh():
    i = np.arange(300, dtype=float)
    x = np.sin(i * 0.37)
    m = 0.8 * x + np.cos(i * 0.91)
    y = 1.7 * x + 4.0 * m + 0.01 * np.sin(i * 0.23)
    data = {"x": x, "m": m, "y": y}
    dag = ac.Dag.from_edges(list(data), [("x", "m"), ("x", "y"), ("m", "y")])
    query = ac.NestedCounterfactual("x", "m", "y", control_level=-1.0, active_level=2.0)
    identified = identify(graph=dag, query=query)
    assert identified.identifier == "path_specific.natural"

    prepared = _prepare(data, graph=dag, query=query, refute="none")
    outcome = ac.estimate(prepared)
    result = require_success(outcome)
    assert result.effect == pytest.approx(5.1, abs=1e-3)
    assert result.estimate.estimator_id == "mediation.linear"
    assert any("counterfactual.nested.shared_exogenous" in d for d in result.diagnostics)
    assert result.estimate.se_bootstrap is None

    query_artifact = artifacts.loads(bytes(outcome.export(quantity="query")))
    assert query_artifact.payload_kind == "query"
    assert "nested_counterfactual" in str(query_artifact.payload).lower()
    result_artifact = artifacts.loads(bytes(outcome.export(quantity="record")))
    assert result_artifact.payload["estimate"] == pytest.approx(result.effect)
    assert prepared.evidence_status == "licensed"
    checked_receipt = artifacts.accept(bytes(result.export()))
    assert checked_receipt["accepts_as_verified_program"] == "true"
    assert checked_receipt["unresolved"] == ""

    refreshed_data = {"x": x, "m": m + 0.2, "y": y + 0.7}
    refreshed = require_success(ac.estimate(ac.rebind(prepared, data=refreshed_data)))
    assert refreshed.effect == pytest.approx(result.effect, abs=1e-8)
    refreshed_receipt = artifacts.accept(bytes(refreshed.export()))
    assert refreshed_receipt["accepts_as_verified_program"] == "true"
    assert refreshed_receipt["program"] == checked_receipt["program"]


@pytest.mark.parametrize(
    "kind,pin_key",
    [
        ("point", "point"),
        ("elasticity", "elasticity"),
        ("semi", "semi_treatment"),
        ("semi_outcome", "semi_outcome"),
        ("average", "average"),
        ("jacobian", "jacobian"),
        ("directional", "directional"),
    ],
)
@pytest.mark.parametrize("accepted", [False, True])
def test_derivative_stages_and_artifacts(kind, pin_key, accepted):
    pin = load_json(ROOT / "conformance/response/staged_derivatives/expected.json")
    i = np.arange(800, dtype=float)
    a = 2 + np.sin(i * 0.71) + 0.2 * np.cos(i * 0.13)
    b = np.cos(i * 1.13)
    data = {"a": a, "b": b, "y": 5 + 2 * a - 0.5 * b, "v": 1 + 0.25 * a + 1.5 * b}
    graph = ac.Dag.from_edges(list(data), [("a", "y"), ("b", "y"), ("a", "v"), ("b", "v")])
    graph = ac.AcceptedGraph(graph) if accepted else graph
    query = {
        "point": ac.PointDerivative("a", "y", at=2),
        "elasticity": ac.Elasticity("a", "y", at=2),
        "semi": ac.SemiElasticity("a", "y", at=2),
        "semi_outcome": ac.SemiElasticity("a", "y", at=2, log_scale="outcome"),
        "average": ac.AverageDerivative("a", "y"),
        "jacobian": ac.ResponseJacobian(["a", "b"], ["y", "v"], at=[2, 0]),
        "directional": ac.DirectionalDerivative(
            ["a", "b"], ["y", "v"], at=[2, 0], direction=[1, 2]
        ),
    }[kind]
    assert identify(graph=graph, query=query)
    config = (
        {"bandwidth": 0.35} if kind in {"point", "elasticity", "semi", "semi_outcome"} else None
    )
    prepared = _prepare(data, graph=graph, query=query, estimator_config=config)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    outcome = ac.estimate(prepared)
    result = require_success(outcome)
    wire = artifacts.loads(bytes(outcome.export(quantity="record")))
    assert wire.payload_kind == "response_result"
    assert result.identification
    got = np.asarray(result.estimate, dtype=float).ravel()
    want = np.asarray(pin[pin_key], dtype=float).ravel()
    assert got == pytest.approx(want, abs=pin["tolerance"])
    fresh = require_success(
        ac.analyze(data, graph=graph, query=query, estimator_config=config, refute="none")
    )
    np.testing.assert_array_equal(np.asarray(fresh.estimate, dtype=float).ravel(), got)
    # cheap/full run the derivative refuter state, which is unlicensed.
    refuted = require_success(
        ac.analyze(data, graph=graph, query=query, estimator_config=config, refute="cheap")
    )
    assert refuted.evidence_status != "licensed"
    assert refuted.validation is not None
    checks = [check.id for check in refuted.validation.checks]
    assert "derivative.support" in checks, checks
    assert np.asarray(refuted.estimate, dtype=float).ravel() == pytest.approx(
        want, abs=pin["tolerance"]
    )


@pytest.mark.parametrize("kind", ["jacobian", "directional"])
def test_frequentist_plugin_gradient_is_point_only_and_says_why(kind):
    """The Frequentist Jacobian / directional derivative publishes no interval.

    Its coefficient-sandwich band is wired for calibration but stays closed until
    its coverage is measured; the warning names that, and Bayesian inference is
    the route with a band.
    """
    rng = np.random.default_rng(23)
    x = rng.normal(size=400)
    a = 0.5 * x + rng.normal(size=400)
    b = 0.3 * x + rng.normal(size=400)
    data = {
        "a": a,
        "b": b,
        "x": x,
        "y": 1 + 2 * a - 0.5 * b + x + rng.normal(size=400),
        "v": 0.25 * a + 1.5 * b + 0.5 * x + rng.normal(size=400),
    }
    graph = ac.Dag.from_edges(
        list(data),
        [("x", "a"), ("x", "b"), ("x", "y"), ("x", "v")]
        + [(t, o) for t in ("a", "b") for o in ("y", "v")],
    )
    query = {
        "jacobian": ac.ResponseJacobian(["a", "b"], ["y", "v"], at=[0.5, 0.0]),
        "directional": ac.DirectionalDerivative(
            ["a", "b"], ["y", "v"], at=[0.5, 0.0], direction=[1, 2]
        ),
    }[kind]
    result = require_success(
        ac.analyze(data, graph=graph, query=query, refute="none", execution=ac.Execution(seed=3))
    )
    assert np.all(np.isfinite(np.asarray(result.estimate, dtype=float)))
    assert result.uncertainty.kind == "none"
    assert result.uncertainty.lower is None and result.uncertainty.upper is None
    withheld = [w for w in result.support.warnings if "response.derivative_interval_withheld" in w]
    assert withheld, result.support.warnings
    assert "not yet measured" in withheld[0]
    assert "Bayesian" in withheld[0]


@pytest.mark.parametrize("kind", ["selected", "right_km", "left_km", "right_cox", "left_cox"])
@pytest.mark.parametrize("accepted", [False, True])
def test_observation_pair_fixture(kind, accepted):
    from antecedent import observation as obs

    pin = load_json(ROOT / f"conformance/response/observation_pairs/{kind}.json")
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
    prepared = _prepare(data, graph=graph, query=query)
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    outcome = ac.estimate(prepared)
    result = require_success(outcome)
    fresh = require_success(ac.analyze(data, graph=graph, query=query, refute="none"))
    assert fresh.response.values == result.response.values
    assert np.asarray(result.response.values).ravel() == pytest.approx(
        np.array(pin["mean"]) * sign, abs=pin["atol"]
    )
    assert result.uncertainty.kind == "none"
    wire = artifacts.loads(bytes(outcome.export(quantity="record")))
    assert wire.payload_kind == "response_result"
    assert wire.payload["assumptions"]


def test_static_mediation_refuter_pin():
    pin = load_json(ROOT / "conformance/estimate/staged_static_kinds/refuters.json")
    data, graph = fixture()
    query = ac.MediationEffect(
        "a", "y", mediators=["m"], contrast="natural_indirect", control_level=0.2, active_level=0.8
    )
    prepared = _prepare(data, graph=graph, query=query, refute="full", bootstrap=0)
    outcome = ac.estimate(ac.reconfigure(prepared, execution=ac.Execution(seed=pin["seed"])))
    result = require_success(outcome)
    for report, expected in zip(result.validation.reports, pin["reports"], strict=True):
        actual = report.to_dict()
        for key in ("original_ate", "refuted_ate", "comparison"):
            got = actual.pop(key)
            assert got == pytest.approx(expected[key], abs=pin["atol"])
        assert actual == {k: v for k, v in expected.items() if k in actual}


def test_new_derivative_boundaries():
    with pytest.raises(ValueError, match="Elasticity.at must be positive"):
        ac.Elasticity("a", "y", at=-0.5)
    data, graph = fixture()
    for query, config, error, message in [
        (
            ac.PointDerivative("a", "y", at=0.5),
            None,
            ac.errors.CausalEstimateError,
            "point derivatives require an explicit bandwidth",
        ),
        (
            ac.PointDerivative("a", "y", at=0.5),
            {"bandwidth": -0.2},
            ac.errors.CausalEstimateError,
            "invalid continuous-response options",
        ),
        (
            ac.AverageDerivative("a", "y", weighting="custom"),
            None,
            ValueError,
            "the current ADE estimator supports weighting='observed'",
        ),
        (
            ac.ResponseJacobian(["a", "m", "y"], ["y"], at=[0, 0, 0]),
            None,
            ac.errors.CausalCompileError,
            "Jacobian dimensions/values are inconsistent",
        ),
    ]:
        with pytest.raises(error, match=message):
            prepared = _prepare(data, graph=graph, query=query, estimator_config=config)
            outcome = ac.estimate(prepared)
            require_success(outcome)


@pytest.mark.parametrize("kind,pin_key", [("point", "point"), ("average", "average")])
def test_confounded_derivatives_are_not_the_observational_slope(kind, pin_key):
    pin = load_json(ROOT / "conformance/response/staged_derivatives/confounded.json")
    data, graph = confounded_derivatives()
    naive = np.linalg.lstsq(
        np.column_stack([np.ones(len(data["a"])), data["a"]]), data["y"], rcond=None
    )[0][1]
    structural = pin["point"]
    assert abs(naive - structural) > pin["naive_gap_min"]
    query = {
        "point": ac.PointDerivative("a", "y", at=2),
        "average": ac.AverageDerivative("a", "y"),
    }[kind]
    config = {"bandwidth": 0.35} if kind == "point" else None
    result = require_success(_estimate(data, graph=graph, query=query, estimator_config=config))
    got = float(np.asarray(result.estimate, dtype=float).ravel()[0])
    assert got == pytest.approx(pin[pin_key], abs=pin["tolerance"])
    assert abs(got - structural) * 4 < abs(naive - structural)


@pytest.mark.parametrize(
    "contrast,key",
    [("natural_direct", "direct"), ("natural_indirect", "indirect"), ("total", "total")],
)
def test_confounded_mediation_is_not_the_unadjusted_association(contrast, key):
    pin, data, graph = confounded_static()
    intercept = np.ones(len(data["a"]))
    ba, bm = np.linalg.lstsq(
        np.column_stack([intercept, data["a"], data["m"]]), data["y"], rcond=None
    )[0][1:]
    naive = (
        ba * (pin["active"] - pin["control"])
        if key == "direct"
        else bm * 2 * (pin["active"] - pin["control"])
    )
    if key == "total":
        naive = np.linalg.lstsq(np.column_stack([intercept, data["a"]]), data["y"], rcond=None)[0][
            1
        ] * (pin["active"] - pin["control"])
    assert abs(naive - pin[key]) > pin["tolerance"]
    q = ac.MediationEffect(
        "a",
        "y",
        mediators=["m"],
        contrast=contrast,
        control_level=pin["control"],
        active_level=pin["active"],
    )
    result = require_success(_estimate(data, graph=graph, query=q, bootstrap=0))
    assert result.effect == pytest.approx(pin[key], abs=pin["tolerance"])
    assert result.estimate.estimator_id == "mediation.linear"


def test_confounded_counterfactual_is_not_the_observational_slope():
    pin, data, graph = confounded_static()
    naive = np.linalg.lstsq(
        np.column_stack([np.ones(len(data["a"])), data["a"]]), data["y"], rcond=None
    )[0][1] * (pin["active"] - pin["control"])
    assert abs(naive - pin["counterfactual_mean"]) > pin["tolerance"]
    q = ac.Counterfactual("a", "y", control_level=pin["control"], active_level=pin["active"])
    result = require_success(_estimate(data, graph=graph, query=q, refute="none"))
    assert result.effect == pytest.approx(pin["counterfactual_mean"], abs=pin["tolerance"])
    assert result.estimate.estimator_id == ac.Estimator.GCM_FIT
    assert abs(result.mean_ite - naive) > pin["tolerance"]


def test_static_artifact_rejects_wrong_family_payload():
    data, graph = fixture()
    prepared = _prepare(data, graph=graph, query=ac.Counterfactual("a", "y"))
    outcome = ac.estimate(prepared)
    require_success(outcome)
    wire = artifacts.loads(bytes(outcome.export(quantity="record")))
    payload = dict(wire.payload)
    payload["unit_effects"] = None
    with pytest.raises(ac.errors.CausalSerializationError, match="does not match query family"):
        artifacts.dumps(
            "static_result", payload, variable_names=wire.variable_names, artifact_id="bad"
        )


@pytest.mark.parametrize("left", [False, True])
def test_python_consumes_cox_oracle(left):
    from antecedent import observation as obs

    pin = load_json(ROOT / "conformance/response/conditional_ipcw/expected.json")
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
    assert adjusted.values == pytest.approx(
        np.array(data["y"]) * np.array(adjusted.weights), abs=pin["atol"]
    )


def test_counterfactual_prepared_refits_supplied_data():
    data, graph = fixture()
    query = ac.Counterfactual("a", "y", control_level=0.2, active_level=0.8)
    prepared = _prepare(data, graph=graph, query=query)
    outcome = ac.estimate(prepared)
    original = require_success(outcome)
    changed = {**data, "y": 2 * data["y"]}
    refitted = require_success(ac.estimate(ac.rebind(prepared, data=changed)))
    fresh = require_success(ac.analyze(changed, graph=graph, query=query, refute="none"))
    assert refitted.effect == pytest.approx(2 * original.effect)
    assert refitted.unit_effects == pytest.approx(fresh.unit_effects)
    assert refitted.identification and refitted.assumptions and refitted.support
    assert np.isnan(refitted.estimate.se_analytic)
    assert refitted.estimate.se_bootstrap is None
    assert any("reused" in d for d in refitted.diagnostics)


def test_static_mediation_bootstrap_is_labelled_as_sampling_uncertainty():
    data, graph = fixture()
    query = ac.MediationEffect("a", "y", mediators=["m"], contrast="natural_direct")
    prepared = _prepare(data, graph=graph, query=query, bootstrap=20)
    outcome = ac.estimate(prepared)
    result = require_success(outcome)
    assert np.isnan(result.estimate.se_analytic)
    assert result.estimate.se_bootstrap > 0
    wire = artifacts.loads(bytes(outcome.export(quantity="record")))
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
    with pytest.raises(ac.CausalError) as caught:
        require_success(_estimate(data, graph=graph, query=query))
    assert_registered_refusal(caught.value)


def test_v13_named_refusals():
    data, dag = fixture()
    posterior = ac.discovery.ExactDagPosterior().run(data, seed=1)
    cf = ac.Counterfactual("a", "y", control_level=0.2, active_level=0.8)
    med = ac.MediationEffect("a", "y", mediators=["m"], contrast="natural_direct")
    accepted_cf = require_success(
        ac.analyze(data, graph=ac.AcceptedGraph.from_graph(dag), query=cf, refute="none")
    )
    assert accepted_cf.effect == pytest.approx(
        require_success(ac.analyze(data, graph=dag, query=cf, refute="none")).effect
    )
    # Cheap validation of a counterfactual on a supplied Dag runs the functional refuter state
    # and stays unlicensed (no evidence row or coverage record names the cell).
    validated_cf = require_success(ac.analyze(data, graph=dag, query=cf, refute="cheap"))
    assert validated_cf.evidence_status != "licensed"
    med_bayes = require_success(
        ac.analyze(data, graph=dag, query=med, inference=ac.Bayesian(n_draws=64), refute="none")
    )
    assert med_bayes.posterior is not None
    cf_bayes = require_success(
        ac.analyze(data, graph=dag, query=cf, inference=ac.Bayesian(n_draws=32), refute="none")
    )
    assert cf_bayes.posterior is not None
    deriv = require_success(
        ac.analyze(
            data,
            graph=dag,
            query=ac.PointDerivative("a", "y", at=0.5),
            inference=ac.Bayesian(n_draws=64),
            estimator_config={"bandwidth": 0.35},
            refute="none",
        )
    )
    assert deriv.response is not None or deriv.ate is not None
    # A counterfactual has no graph-posterior mixture in the facade; a mediation does: every
    # atom of the posterior decomposes it with the executor of its own structure and the
    # atoms are mixed by posterior probability.
    with pytest.raises(CausalUnsupportedError, match="no graph-posterior mixture"):
        require_success(
            ac.analyze(
                data,
                query=cf,
                discovery=posterior,
                inference=ac.Bayesian(),
                refute="none",
            )
        )
    mixed = require_success(
        ac.analyze(
            data,
            query=med,
            discovery=posterior,
            inference=ac.Bayesian(n_draws=64),
            refute="none",
        )
    )
    assert np.isfinite(mixed.effect)
    assert mixed.evidence_status != "licensed"
    # Full validation and a prepared cheap suite run the same functional refuter state.
    assert (
        require_success(ac.analyze(data, graph=dag, query=cf, refute="full")).evidence_status
        != "licensed"
    )
    prepared_cf = _prepare(data, graph=dag, query=cf, refute="cheap")
    assert require_success(prepared_cf.estimate()).evidence_status != "licensed"
    # A Frequentist replicate count runs the mechanism-refit row bootstrap, which is
    # unlicensed; the Bayesian cell keeps its posterior and refuses bootstrap=.
    bootstrapped = require_success(
        ac.analyze(data, graph=dag, query=cf, refute="none", inference=ac.Frequentist(bootstrap=10))
    )
    assert bootstrapped.evidence_status != "licensed"
    assert any("gcm.frequentist_bootstrap" in d for d in bootstrapped.diagnostics)
    # The Bayesian cell keeps its posterior: its declaration has no replicate budget at all.
    with pytest.raises(TypeError, match="bootstrap"):
        ac.Bayesian(bootstrap=8)  # type: ignore[call-arg]


def test_counterfactual_reports_extrapolative_support():
    data, graph = fixture()
    query = ac.Counterfactual("a", "y", control_level=-4.0, active_level=4.0)
    result = require_success(_estimate(data, graph=graph, query=query, refute="none"))
    assert any("extrapolative=true" in item for item in result.support)
