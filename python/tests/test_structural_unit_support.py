"""Each structural program reports the support decision of the unit its route runs under."""

import random

from antecedent import _native, events, latent_soft_intervention, mnar_fixed_shift


def _assert_decision(decision: dict | None, unit: str) -> None:
    assert decision is not None
    assert decision["unit"] == unit
    assert decision["supported"] == (decision["status"] == "supported")
    assert decision["status"] in {"supported", "not_supported"}
    if decision["supported"]:
        assert decision["missing_evidence"] == []


def test_event_process_reports_its_unit() -> None:
    declaration = {
        "operation": "forecast",
        "request": {
            "history": {
                "catalog": ["mention"],
                "time_unit": "hour",
                "origin": "launch",
                "source": "news",
                "empty_origin": 0.0,
                "window": {"start": 0.0, "end": 1.0},
                "left": [],
                "events": [],
            },
            "forecast": {"start": 1.0, "end": 3.0},
            "source": {"kind": "fixture", "paths": [[{"time": 1.25, "kind": 0}], []]},
            "paths": 2,
            "max_events": 8,
            "seed": 33,
        },
    }
    _assert_decision(events.prepare(declaration).unit_support(), "forecast.event_process")


def test_latent_soft_intervention_reports_its_unit() -> None:
    rng = random.Random(230)
    groups: list[list[list[float]]] = [[], []]
    for arm, group in enumerate(groups):
        for _ in range(240):
            x = rng.gauss(0.0, 1.0)
            first = 0.3 * x + rng.gauss(0.0, 0.5) + 2.0 * arm
            second = -0.3 * x + rng.gauss(0.0, 0.5)
            group.append([x, first, second, first + second + rng.gauss(0.0, 0.5)])
    prepared = latent_soft_intervention.prepare(
        variables=("x", "first", "second", "outcome"),
        edges=(("x", "first"), ("x", "second"), ("first", "outcome"), ("second", "outcome")),
        context=("x",),
        candidates=("first", "second"),
        baseline=groups[0],
        action=groups[1],
        population="path_population",
        randomized=True,
        version_exogenous_given_context=True,
    )
    _assert_decision(prepared.unit_support(), "estimate.latent_soft_intervention")


def test_mnar_fixed_shift_reports_its_unit() -> None:
    ids: list[int] = []
    arm: list[float] = []
    outcome: list[float | None] = []
    for flag, (assigned, observed) in ((0.0, (30, 24)), (1.0, (40, 28))):
        for i in range(assigned):
            ids.append(len(ids))
            arm.append(flag)
            outcome.append((i * 7 % 11) / 3.0 + flag if i < observed else None)
    prepared = mnar_fixed_shift.prepare_fixed_shift(
        ids,
        arm,
        outcome,
        population="target",
        design="independent_randomized_arms",
        treated_shift=0.8,
    )
    _assert_decision(prepared.unit_support(), "mnar_fixed_shift.compiled_candidate")


def test_the_native_handles_expose_the_decision() -> None:
    for name in (
        "NativePreparedEventProgram",
        "NativePreparedLatentSoftIntervention",
        "NativePreparedMnarFixedShift",
    ):
        assert callable(getattr(_native, name).unit_support)
