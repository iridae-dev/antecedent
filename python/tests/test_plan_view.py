"""The prepared program's plan view, end to end from Python.

``prepared.plan_view()`` reads every layer of the lowered program before any
estimate, names the sealed operation preparation retained, and does not execute:
the handle still has no result after reading it, and ``inspect()`` keeps working.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.estimation import ProgramPlanView
from antecedent.outcomes import require_success


def _table(n: int = 240) -> dict[str, np.ndarray]:
    z = np.array([math.sin(i / 17.0) for i in range(n)])
    t = (z + np.array([math.cos(i / 11.0) for i in range(n)]) > 0).astype(float)
    y = 1.0 + 2.0 * t + 0.8 * z
    return {"t": t, "y": y, "z": z}


_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


def _prepared(**options):
    return ant.prepare(
        _table(),
        graph=_EDGES,
        query=ant.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
        seed=3,
        **options,
    )


def test_plan_view_reads_every_layer_before_any_estimate() -> None:
    prepared = _prepared()
    view = prepared.plan_view()
    assert isinstance(view, ProgramPlanView)
    assert prepared.last_result is None, "reading the view executes nothing"
    assert view.route == "StaticAteDag"
    assert view.family == "static_effect"
    assert view.stage == "prepared"
    assert view.depth == "complete"
    assert len(view.identity) == 64

    assert view.scientific.plan_id
    assert set(view.scientific.query_variables) >= {"t", "y"}

    identification = view.identification
    assert identification.kind == "retained"
    assert identification.status == "nonparametrically_identified"
    assert identification.installed_at_prepare is True
    static = identification.retained[0]
    assert static["kind"] == "static"
    assert list(static["estimands"][0]["adjustment_set"]) == ["z"]
    assert static["assumptions"], "the assumptions identification relies on are listed"
    assert identification.unresolved_obligations == []

    assert view.statistical.estimator
    assert view.statistical.realization == "frequentist"
    assert view.evidence.composition["kind"] == "single"
    assert view.uncertainty.reading == "sampling"
    assert view.physical.record is not None
    assert view.physical.record["estimated_peak_memory_bytes"] > 0
    assert (view.physical.quoted_width is None) == (view.physical.not_design_based is not None)
    assert view.support.licensed is (view.support.status == "licensed")
    assert view.support.status in ("licensed", "allowed_unlicensed")
    assert view.mixture is None
    assert view.operation is not None and view.operation["kind"]

    again = prepared.plan_view()
    assert again.identity == view.identity, "the view of one handle is stable"


def test_plan_view_identity_is_the_executed_program() -> None:
    prepared = _prepared()
    view = prepared.plan_view()
    result = require_success(prepared.estimate(_table(), seed=3))
    assert result.ate == pytest.approx(2.0, abs=0.2)
    assert prepared.plan_view().identity == view.identity
    refreshed = require_success(prepared.refresh({**_table(), "y": _table()["y"] + 0.5}, seed=3))
    assert refreshed.ate == pytest.approx(2.0, abs=0.2)
    assert prepared.plan_view().identity == view.identity, "a refresh keeps the program"


def test_plan_view_to_dict_is_json_safe_and_inspect_still_works() -> None:
    import json

    prepared = _prepared()
    payload = prepared.plan_view().to_dict()
    json.dumps(payload)
    for layer in (
        "scientific",
        "identification",
        "statistical",
        "evidence",
        "physical",
        "uncertainty",
        "support",
    ):
        assert layer in payload, layer
    assert payload["identity"] == prepared.plan_view().identity
    report = prepared.inspect()
    assert report.stage == "prepared"
    preflight = prepared.preflight()
    assert preflight is not None


def test_bayesian_plan_view_states_its_posterior_plan() -> None:
    prepared = _prepared(inference=ant.Bayesian(backend="conjugate", n_draws=64, prior_scale=30.0))
    view = prepared.plan_view()
    assert view.uncertainty.reading == "posterior"
    assert view.statistical.posterior is not None
    assert view.statistical.posterior["draws"] == 64
    assert view.operation["kind"] == "bayesian_gcomp"
    assert view.operation["estimator"] == "bayesian.gcomp"
