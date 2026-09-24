"""The compiler surface says the same thing everywhere it is read.

- An estimator never runs under an inference mode it does not implement; the
  request is refused with a registered code instead of executing the other mode
  under the requested license coordinate.
- A refusal reaching Python carries a registered reason code.
- A not-identified question is a typed refusal carrying its identification
  outcome.
- A study's cheap preflight reports no program identity it has not compiled.
- A live report and the report of its loaded export agree: one ``contract``
  shape, and the same identification, validation and unit-effect evidence.
- Sharp RD's contract names its per-execution identification.
- A retarget preview reports the refusal the retarget raises.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native
from antecedent.ids import Estimator

REGISTERED = frozenset(_native.runtime_refusal_codes())
GRAPH = [("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")]


def _static(seed: int, n: int = 600) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.5 * t + z + rng.normal(size=n)
    return {"z": z, "treatment": t, "outcome": y}


def _binary_outcome(seed: int, n: int = 600) -> dict[str, np.ndarray]:
    data = _static(seed, n)
    data["outcome"] = (data["outcome"] > np.median(data["outcome"])).astype(float)
    return data


def _rd(seed: int, n: int = 2000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    t = (x >= 0).astype(float)
    return {"x": x, "t": t, "y": 2 * t + x + 0.5 * rng.normal(size=n)}


RD_GRAPH = [("x", "t"), ("x", "y"), ("t", "y")]
RD_OPTIONS = dict(identifier="rd.sharp", running_variable="x", cutoff=0.0, bandwidth=0.3)


def _cpdag() -> Any:
    return ant.Cpdag.from_directed_undirected(
        ["z", "treatment", "outcome"],
        [("z", "treatment"), ("z", "outcome")],
        [("treatment", "outcome")],
    )


def _admg() -> Any:
    return ant.Admg.from_edges(["z", "treatment", "outcome"], GRAPH, [])


def _iv(seed: int, n: int = 800) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z, u = rng.normal(size=n), rng.normal(size=n)
    t = (0.8 * z + 0.5 * u + 0.3 * rng.normal(size=n) > 0).astype(float)
    return {"t": t, "y": 1.5 * t + u + 0.3 * rng.normal(size=n), "z": z}


def _frontdoor(seed: int, n: int = 800) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    m = t + 0.4 * rng.normal(size=n)
    return {"t": t, "m": m, "y": m + 0.4 * rng.normal(size=n)}


def test_prepared_aipw_refresh_and_independent_artifact_consumption() -> None:
    """The Python entry point retains the checked AIPW row binding across clicks."""
    data = _static(27, 512)
    graph = GRAPH
    prepared = ant.prepare(
        data,
        graph=graph,
        query=ant.AverageEffect("treatment", "outcome"),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    first = prepared.estimate(data)
    assert first.effect == pytest.approx(1.5, abs=0.3)
    accepted = ant.artifacts.accept(first.export())
    assert accepted["accepts_as_verified_program"] == "true"

    changed = {**data, "outcome": data["outcome"] + 0.75 * data["treatment"]}
    refreshed = prepared.refresh(changed)
    assert refreshed.effect == pytest.approx(first.effect + 0.75, abs=0.05)
    accepted_refresh = ant.artifacts.accept(refreshed.export())
    assert accepted_refresh["accepts_as_verified_program"] == "true"
    assert accepted_refresh["program"] == accepted["program"]
    assert refreshed.data_snapshot_id != first.data_snapshot_id


def test_prepared_frontdoor_refresh_and_independent_artifact_consumption() -> None:
    """A checked front-door result survives public refresh and portable consumption."""
    data = _frontdoor(31)
    prepared = ant.prepare(
        data,
        graph=[("t", "m"), ("m", "y")],
        query=ant.AverageEffect("t", "y"),
        identifier="frontdoor",
        estimator="frontdoor.linear_two_stage",
        refute="none",
        bootstrap=0,
    )
    first = prepared.estimate(data)
    assert first.effect == pytest.approx(1.0, abs=0.15)
    accepted = ant.artifacts.accept(first.export())
    assert accepted["accepts_as_verified_program"] == "true"

    changed = {**data, "y": data["y"] + 0.4}
    refreshed = prepared.refresh(changed)
    assert refreshed.effect == pytest.approx(first.effect, abs=1e-10)
    accepted_refresh = ant.artifacts.accept(refreshed.export())
    assert accepted_refresh["accepts_as_verified_program"] == "true"
    assert accepted_refresh["program"] == accepted["program"]
    assert refreshed.data_snapshot_id != first.data_snapshot_id


# (name, data, graph, query, extra analyze kwargs)
ESTIMATOR_ROUTES: list[tuple[str, Callable[[], Any], Any, Any, dict[str, Any]]] = [
    ("dag", lambda: _static(1), GRAPH, ant.AverageEffect("treatment", "outcome"), {}),
    (
        "dag_binary",
        lambda: _binary_outcome(1),
        GRAPH,
        ant.AverageEffect("treatment", "outcome"),
        {},
    ),
    ("cpdag", lambda: _static(1), _cpdag(), ant.AverageEffect("treatment", "outcome"), {}),
    ("admg", lambda: _static(1), _admg(), ant.AverageEffect("treatment", "outcome"), {}),
    (
        "iv",
        lambda: _iv(1),
        [("z", "t"), ("t", "y")],
        ant.AverageEffect("t", "y"),
        {"identifier": "iv"},
    ),
    (
        "frontdoor",
        lambda: _frontdoor(1),
        [("t", "m"), ("m", "y")],
        ant.AverageEffect("t", "y"),
        {"identifier": "frontdoor"},
    ),
    ("rd", lambda: _rd(1), RD_GRAPH, ant.AverageEffect("t", "y"), RD_OPTIONS),
    (
        "conditional",
        lambda: _static(1),
        GRAPH,
        ant.ConditionalEffect("treatment", "outcome", modifier="z"),
        {},
    ),
    (
        "mediation",
        lambda: {**_static(1), "m": _static(2)["z"]},
        GRAPH + [("treatment", "m"), ("m", "outcome")],
        ant.MediationEffect("treatment", "outcome", mediators=["m"]),
        {},
    ),
    ("counterfactual", lambda: _static(1), GRAPH, ant.Counterfactual("treatment", "outcome"), {}),
]


@pytest.mark.parametrize(
    "name, make, graph, query, extra", ESTIMATOR_ROUTES, ids=[r[0] for r in ESTIMATOR_ROUTES]
)
def test_no_estimator_runs_under_an_inference_mode_it_does_not_implement(
    name: str, make: Callable[[], Any], graph: Any, query: Any, extra: dict[str, Any]
) -> None:
    """Every estimator override × inference mode either refuses or runs the
    requested mode: a Bayesian request returns a posterior and a Frequentist
    request does not. ``estimator_inference_mismatch`` is the refusal when the
    estimator implements only the other mode."""
    data = make()
    silent: list[str] = []
    ran: list[str] = []
    mismatches = 0
    for estimator in [None, *(member.value for member in Estimator)]:
        for inference in (ant.Frequentist(), ant.Bayesian(n_draws=32)):
            bayesian = isinstance(inference, ant.Bayesian)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ant.analyze(
                        data,
                        graph=graph,
                        query=query,
                        inference=inference,
                        estimator=estimator,
                        refute="none",
                        seed=1,
                        **extra,
                    )
            except Exception as error:  # noqa: BLE001 - every refusal is acceptable here
                code = getattr(error, "reason_code", None)
                if code == "estimator_inference_mismatch":
                    mismatches += 1
                continue
            ran.append(f"{estimator} under {type(inference).__name__}")
            if (result.posterior is not None) != bayesian:
                silent.append(ran[-1])
    assert not silent, f"{name}: ran a different inference mode than requested: {silent}"
    # The route is exercised: at least one Frequentist execution ran.
    assert any(entry.endswith("Frequentist") for entry in ran), (name, ran)
    if name in ("dag", "dag_binary", "cpdag", "admg", "iv", "frontdoor", "rd"):
        assert mismatches > 0


def test_bayesian_request_with_a_frequentist_estimator_is_refused_by_code() -> None:
    for estimator in ("aipw", "linear.adjustment.ate", "propensity.weighting"):
        with pytest.raises(ant.errors.CausalUnsupportedError) as caught:
            ant.analyze(
                _static(3),
                graph=GRAPH,
                query=ant.AverageEffect("treatment", "outcome"),
                inference=ant.Bayesian(n_draws=32),
                estimator=estimator,
                refute="none",
            )
        assert caught.value.reason_code == "estimator_inference_mismatch"
        message = str(caught.value)
        # The refusal names the estimator and both ways forward; nothing is lost:
        # the estimator still runs under the inference it implements.
        assert f"estimator {estimator} is Frequentist" in message
        assert "Omit estimator=" in message and "inference=Frequentist" in message
        frequentist = ant.analyze(
            _static(3),
            graph=GRAPH,
            query=ant.AverageEffect("treatment", "outcome"),
            estimator=estimator,
            refute="none",
        )
        assert frequentist.answer.kind == "point"
    with pytest.raises(ant.errors.CausalUnsupportedError) as caught:
        ant.prepare(
            _static(3),
            graph=GRAPH,
            query=ant.AverageEffect("treatment", "outcome"),
            estimator="bayesian.gcomp",
        )
    assert caught.value.reason_code == "estimator_inference_mismatch"


# --- reason codes on refusals ------------------------------------------------------


def _refusals() -> list[tuple[str, Callable[[], Any]]]:
    data = _static(0, 400)
    n = len(data["z"])
    query = ant.AverageEffect("treatment", "outcome")
    linear = ant.prepare(data, graph=GRAPH, query=query)
    return [
        (
            "ResponseCurve refute=full",
            lambda: ant.analyze(
                data,
                graph=GRAPH,
                query=ant.ResponseCurve("z", "outcome", grid=[0.0, 1.0]),
                refute="full",
            ),
        ),
        (
            "Counterfactual refute=cheap",
            lambda: ant.analyze(
                data, graph=GRAPH, query=ant.Counterfactual("treatment", "outcome"), refute="cheap"
            ),
        ),
        (
            "unknown estimator",
            lambda: ant.analyze(data, graph=GRAPH, query=query, estimator="bogus"),
        ),
        ("RD kwarg off route", lambda: ant.analyze(data, graph=GRAPH, query=query, bandwidth=0.3)),
        (
            "regimes without RPCMCI",
            lambda: ant.analyze(data, graph=GRAPH, query=query, regimes=[0] * n),
        ),
        (
            "unknown variable",
            lambda: ant.analyze(data, graph=GRAPH, query=ant.AverageEffect("treatment", "zz")),
        ),
        (
            "graph cycle",
            lambda: ant.analyze(data, graph=GRAPH + [("outcome", "z")], query=query),
        ),
        ("unknown keyword", lambda: ant.analyze(data, graph=GRAPH, query=query, estimater="aipw")),
        ("refute=True", lambda: ant.analyze(data, graph=GRAPH, query=query, refute=True)),
        (
            "Pulse on static edges",
            lambda: ant.analyze(
                data, graph=GRAPH, query=ant.PulseEffect("treatment", "outcome", horizon_steps=1)
            ),
        ),
        (
            "retarget on Bayesian",
            lambda: ant.prepare(
                data, graph=GRAPH, query=query, inference=ant.Bayesian(n_draws=32)
            ).retarget(np.ones(n), []),
        ),
        ("retarget without scores", lambda: linear.retarget(np.linspace(1, 2, n), [])),
        (
            "refresh with a missing column",
            lambda: linear.refresh({"z": data["z"], "treatment": data["treatment"]}),
        ),
        ("refresh with an extra column", lambda: linear.refresh({**data, "w": data["z"]})),
        ("estimator ipw", lambda: ant.analyze(data, graph=GRAPH, query=query, estimator="ipw")),
        (
            "identifier frontdoor",
            lambda: ant.analyze(data, graph=GRAPH, query=query, identifier="frontdoor"),
        ),
        (
            "unknown estimator_config key",
            lambda: ant.analyze(data, graph=GRAPH, query=query, estimator_config={"nonsense": 1}),
        ),
        (
            "interactive discovery",
            lambda: ant.analyze(
                data, discovery=ant.discovery.PC(), query=query, latency="interactive"
            ),
        ),
    ]


@pytest.mark.parametrize("label, call", _refusals(), ids=[label for label, _ in _refusals()])
def test_every_probed_refusal_carries_a_registered_reason_code(
    label: str, call: Callable[[], Any]
) -> None:
    with pytest.raises((ant.errors.CausalError, ValueError, TypeError)) as caught:
        call()
    error = caught.value
    assert error.reason_code in REGISTERED, (label, error.reason_code, str(error))
    report = getattr(error, "report", None)
    if report is not None:
        assert report.code == error.reason_code


def test_refusal_report_code_falls_back_when_the_attribute_is_none() -> None:
    from antecedent._api import describe_refusal

    @describe_refusal
    def refuse() -> None:
        raise ant.errors.CausalUnsupportedError("no code on this refusal")

    with pytest.raises(ant.errors.CausalUnsupportedError) as caught:
        refuse()
    assert caught.value.reason_code is None
    assert caught.value.report.code == "CausalUnsupportedError"


# --- not identified ---------------------------------------------------------------


def test_unidentified_question_is_a_typed_refusal_with_its_outcome() -> None:
    bow = ant.Admg.from_edges(["t", "y"], [("t", "y")], [("t", "y")])
    rng = np.random.default_rng(0)
    data = {"t": rng.normal(size=300), "y": rng.normal(size=300)}
    with pytest.raises(ant.errors.EffectNotIdentified) as caught:
        ant.analyze(data, graph=bow, query=ant.AverageEffect("t", "y"))
    error = caught.value
    assert isinstance(error, ant.errors.CausalUnsupportedError)
    assert isinstance(error, ant.errors.CausalCompileError)
    assert error.reason_code == "effect_not_identified"
    assert error.identification_status == "not_identified"
    assert error.search_complete is True and error.search_capped is False
    assert "the search completed" in str(error)


# --- identities, report shape and evidence --------------------------------------------


def test_preflight_reports_no_program_it_has_not_compiled() -> None:
    study = ant.prepare(_static(0), graph=GRAPH, query=ant.AverageEffect("treatment", "outcome"))
    assert study.inspect().program_id is not None
    assert study.preflight().program_id is None
    assert study.estimate().program_id == study.inspect().program_id


def _failing(*, ate: float, se_analytic: float, method: str, adjustment_set: Any) -> dict[str, Any]:
    return {"passed": False, "refuted_ate": 0.0, "comparison": 0.0, "failure_condition": "fails"}


def _evidence(report: dict[str, Any]) -> dict[str, Any]:
    identification = report["identification"]["payload"]
    return {
        "contract_keys": sorted(report["contract"]),
        "method": identification["method"],
        "adjustment_set": identification["adjustment_set"],
        "assumption_count": identification["assumption_count"],
        "derivation_step_count": identification["derivation_step_count"],
        "validation": report["support"]["payload"]["validation"],
        "unit_effects": report["uncertainty"]["payload"].get("unit_effects"),
        "inference_binding": report["inference_binding"],
    }


def test_a_failed_validation_survives_export_and_load() -> None:
    result = ant.analyze(
        _static(4),
        graph=GRAPH,
        query=ant.AverageEffect("treatment", "outcome"),
        refute="none",
        validators=[_failing],
    )
    live = result.inspect().to_dict()
    loaded = ant.load(result.export()).inspect().to_dict()
    assert live["support"]["payload"]["validation"]["passed"] is False
    assert loaded["support"]["payload"]["validation"]["passed"] is False
    assert loaded["support"]["payload"]["validation"]["count"] == 1
    assert _evidence(live) == _evidence(loaded)
    assert live["identification"]["payload"]["adjustment_set"] == ["z"]
    assert live["identification"]["payload"]["method"] == "backdoor.adjustment"


@pytest.mark.parametrize(
    "inference", [ant.Frequentist(), ant.Bayesian(n_draws=64)], ids=["freq", "bayes"]
)
def test_counterfactual_unit_effects_survive_export_and_load(inference: Any) -> None:
    result = ant.analyze(
        _static(5, 200),
        graph=GRAPH,
        query=ant.Counterfactual("treatment", "outcome"),
        inference=inference,
        refute="none",
    )
    live = result.inspect().to_dict()
    loaded = ant.load(result.export()).inspect().to_dict()
    units = loaded["uncertainty"]["payload"]["unit_effects"]
    assert units["effects"] == pytest.approx(list(result.unit_effects))
    assert units["homogeneous"] == result.estimate.unit_effects_homogeneous
    assert _evidence(live) == _evidence(loaded)


def test_live_and_loaded_reports_share_one_contract_shape() -> None:
    result = ant.analyze(
        _static(6), graph=GRAPH, query=ant.AverageEffect("treatment", "outcome"), refute="placebo"
    )
    live = result.inspect().to_dict()
    loaded = ant.load(result.export()).inspect().to_dict()
    json.dumps(live, allow_nan=False)
    assert isinstance(live["contract"], dict) and isinstance(loaded["contract"], dict)
    assert live["contract"] == loaded["contract"]
    assert live["inference_binding"] == loaded["inference_binding"]
    assert isinstance(live["contract"]["identities"], dict)


# --- sharp RD and previews --------------------------------------------------------


def test_sharp_rd_contract_names_per_execution_identification() -> None:
    study = ant.prepare(
        _rd(0),
        graph=RD_GRAPH,
        query=ant.AverageEffect("t", "y"),
        estimator="rd.sharp",
        **RD_OPTIONS,
    )
    identification = study.inspect().identification
    assert identification.available is False
    assert identification.reason == "identified_per_execution"
    assert study.estimate().inspect().identification.summary == "nonparametrically_identified"


def test_retarget_preview_reports_the_refusal_the_retarget_raises() -> None:
    data = _static(7)
    query = ant.AverageEffect("treatment", "outcome")
    n = len(data["z"])
    linear = ant.prepare(data, graph=GRAPH, query=query)
    preview = linear.preview_transform("retarget")
    assert preview["refused"] == "true"
    assert preview["refusal_code"] == "score_table_unavailable"
    with pytest.raises(ant.errors.CausalUnsupportedError) as caught:
        linear.retarget(np.ones(n), [])
    assert caught.value.reason_code == preview["refusal_code"]
    assert str(caught.value) == preview["refusal"]

    aipw = ant.prepare(data, graph=GRAPH, query=query, estimator="aipw")
    preview = aipw.preview_transform("retarget")
    assert preview["refused"] == "false"
    assert "refusal" not in preview
    aipw.retarget(np.ones(n), [])
