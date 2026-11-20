"""Public compiler and artifact dispatch for the latent intervention candidate."""

import random

import pytest

import antecedent as ant
from antecedent import latent_soft_intervention as lsi
from antecedent.errors import CausalError


def _arms() -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(230)
    groups: list[list[list[float]]] = [[], []]
    for arm, group in enumerate(groups):
        for _ in range(240):
            x = rng.gauss(0.0, 1.0)
            first = 0.3 * x + rng.gauss(0.0, 0.5) + 2.0 * arm
            second = -0.3 * x + rng.gauss(0.0, 0.5)
            group.append([x, first, second, first + second + rng.gauss(0.0, 0.5)])
    return groups[0], groups[1]


def test_prepared_forecast_loads_through_shared_dispatch() -> None:
    baseline, action = _arms()
    prepared = lsi.prepare(
        variables=("x", "first", "second", "outcome"),
        edges=(("x", "first"), ("x", "second"), ("first", "outcome"), ("second", "outcome")),
        context=("x",),
        candidates=("first", "second"),
        baseline=baseline,
        action=action,
        population="path_population",
        randomized=True,
        version_exogenous_given_context=True,
    )
    plan = prepared.inspect()
    assert plan["route"] == "LatentSoftIntervention"
    if not plan["support"]["supported"] and ant.build_flavor() == "release":
        with pytest.raises(CausalError) as refused:
            prepared.execute()
        assert refused.value.reason_code == "not_supported"
        return
    fitted = prepared.execute()
    if not plan["support"]["supported"]:
        assert fitted.discovery.would_refuse is not None
    forecast = fitted.forecast({"x": 0.0})
    exported = fitted.export({"x": 0.0})
    consumed = ant.load(exported, expected=exported.expectation)
    assert consumed.acceptance.replayed
    assert consumed.value.forecast == forecast
    with pytest.raises(CausalError):
        ant.load(exported, expected=fitted.export({"x": 0.4}).expectation)
