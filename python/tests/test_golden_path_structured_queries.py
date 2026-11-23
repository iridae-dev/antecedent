"""Golden-path coverage for the graphless / structured query families.

The five-line lifecycle must hold for every query family, not only the graph
routes in ``test_golden_path.py``. Design-carrying queries freeze their design in
the query, so ``rebind`` re-reads outcomes for the *same* rows (``refresh="same"``)
or is refused by construction (``refresh="refused"``); either way the other four
verbs work and, critically, ``load(result.export()).answer == result.answer`` --
the invariant that regressed silently because no case exercised it.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import antecedent as ant
import numpy as np
import pytest
from antecedent import policy, quasi, regimes, survival
from antecedent.errors import CausalUnsupportedError
from antecedent.interference import BernoulliAssignment
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success


def _bernoulli(seed: int, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(size=n) < 0.5
    return {"y": a.astype(float) + rng.normal(size=n), "_assign": a}


def _panel(seed: int, n_units: int = 120) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids, g, post, y = [], [], [], []
    for u in range(n_units):
        gg = u % 2
        for t in (0, 1):
            ids.append(float(u))
            g.append(float(gg))
            post.append(float(t))
            y.append(1.0 + 0.5 * gg + 0.3 * t + 2.0 * gg * t + rng.normal() * 0.2)
    return {"y": np.array(y), "id": np.array(ids), "g": np.array(g), "post": np.array(post)}


def _event(seed: int, n_units: int = 40) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids, per, coh, y = [], [], [], []
    for u in range(n_units):
        c = 0 if u < n_units // 2 else 3
        for t in range(1, 6):
            ids.append(f"s{u}")
            per.append(t)
            coh.append(c)
            y.append(float(u) + (5.0 if (c != 0 and t >= c) else 0.0) + rng.normal() * 0.1)
    return {
        "y": np.array(y, float),
        "id": ids,
        "period": np.array(per, float),
        "cohort": np.array(coh, float),
    }


def _survival(seed: int, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dur = rng.exponential(2.0, n)
    return {
        "duration": dur,
        "event": (dur < 3).astype(float),
        "treated": (rng.uniform(size=n) < 0.5).astype(float),
    }


def _longitudinal(seed: int, n: int = 500) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "y": np.array([4.0 if i % 4 == 0 else 0.0 for i in range(n)]) + rng.normal(size=n) * 0.0
    }


def _policy_query() -> Any:
    n = 400
    a = [i % 2 == 1 for i in range(n)]
    return policy.PolicyValue(
        outcome="y",
        assignment=a,
        propensity=0.5,
        policy=policy.BinaryPolicy([i % 4 < 2 for i in range(n)]),
        evaluation_subject_ids=[f"e{i}" for i in range(n)],
    )


def _randomized_query() -> Any:
    # The same assignment draw as `_bernoulli(1)`, so the design describes the data analysed.
    a = [bool(v) for v in np.random.default_rng(1).uniform(size=400) < 0.5]
    ids = [f"u{i}" for i in range(400)]
    return ant.experiment.RandomizedEffect(
        "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), a, ids, ids)
    )


def _longitudinal_query() -> Any:
    n = 500
    history = [[(i % 4 in (0, 1)), (i % 4 in (0, 2))] for i in range(n)]
    return regimes.LongitudinalRegime(
        outcome="y",
        treatment_history=history,
        actions=[True, True],
        treatment_probabilities=[[0.5, 0.5]] * n,
        subject_ids=[f"subject-{i}" for i in range(n)],
        fold_ids=[i % 5 for i in range(n)],
    )


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[int], dict[str, np.ndarray]]
    query: Any
    kind: str
    refresh: Literal["same", "refused", "skip"] = "same"
    options: dict[str, Any] = field(default_factory=dict)
    # Known-truth check of the numerical answer, from the generating equations of the case.
    truth: Callable[[Any], None] | None = None


def _near(value: float, expected: float, tolerance: float) -> None:
    assert abs(value - expected) < tolerance, f"{value} vs {expected} +- {tolerance}"


def _randomized_truth(result: Any) -> None:
    # y = a + N(0, 1) with a ~ Bernoulli(1/2): the effect is 1; the se is about 0.1 at n = 400.
    _near(result.ate, 1.0, 0.35)


def _policy_truth(result: Any) -> None:
    # The policy treats half the evaluation rows, so its value is E[y(1)] / 2 = 0.5 (se 0.084);
    # the always-control reference is 0 and the treated-everyone reference 1 sit at least 0.5 away.
    _near(result.answer.structured["policy_value"], 0.5, 0.25)


def _did_truth(result: Any) -> None:
    # The interaction coefficient of y = 1 + 0.5 g + 0.3 post + 2 g post + N(0, 0.04) is 2.
    _near(result.answer.structured["effect"], 2.0, 0.2)


def _event_truth(result: Any) -> None:
    # A jump of 5 from the adoption period (cohort 3) and no pre-trend.
    effects = {
        (row["cohort"], row["event_time"]): row["estimate"]
        for row in result.answer.structured["event_time_effects"]
    }
    _near(effects[(3, 0)], 5.0, 0.2)
    _near(effects[(3, -2)], 0.0, 0.2)


def _survival_truth(result: Any) -> None:
    # Exponential durations with mean 2 and no treatment effect: RMST(2) = 2 (1 - exp(-1)).
    expected = 2.0 * (1.0 - math.exp(-1.0))
    structured = result.answer.structured
    _near(structured["rmst_control"], expected, 0.2)
    _near(structured["rmst_treated"], expected, 0.2)


def _longitudinal_truth(result: Any) -> None:
    # Only subjects with history (1, 1) follow the regime; they have y = 4 and no noise.
    assert result.answer.structured["value"] == pytest.approx(4.0, abs=1e-9)
    assert result.answer.structured["matched_observed_fraction"] == pytest.approx(0.25)


CASES = [
    Case("randomized-bernoulli", _bernoulli, _randomized_query(), "point", truth=_randomized_truth),
    Case(
        "policy-value",
        lambda s: {"y": _bernoulli(s)["y"]},
        _policy_query(),
        "policy_value",
        refresh="refused",
        truth=_policy_truth,
    ),
    Case(
        "panel-did",
        _panel,
        quasi.PanelDifferenceInDifferences("y", "id", "g", "post"),
        "structured",
        truth=_did_truth,
    ),
    Case(
        "staggered-event-study",
        _event,
        quasi.StaggeredAdoption("y", "id", "period", "cohort", event_study=True),
        "structured",
        truth=_event_truth,
    ),
    Case(
        "survival-rmst",
        _survival,
        survival.SurvivalOutcome(
            "duration",
            "event",
            "treated",
            tau=2.0,
            randomized=True,
            observation_assumption=IndependentGiven(()),
        ),
        "structured",
        options={"inference": ant.Frequentist(bootstrap=199)},
        truth=_survival_truth,
    ),
    Case(
        "longitudinal-regime",
        _longitudinal,
        _longitudinal_query(),
        "structured",
        truth=_longitudinal_truth,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_path_structured_queries(case: Case) -> None:
    data = case.make(1)
    data = {k: v for k, v in data.items() if not k.startswith("_")}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = require_success(ant.analyze(data, query=case.query, **case.options))

        # 1. answer kind is what the family declares, and its number is the known truth.
        assert result.answer.kind == case.kind
        assert case.truth is not None
        case.truth(result)

        # 2. inspect().to_dict() is JSON and shares its shape with the loaded report.
        report = result.inspect().to_dict()
        json.dumps(report, allow_nan=False)

        # 3. export -> load round-trips, and the loaded answer equals the live answer
        #    (the invariant that silently regressed).
        artifact = result.export()
        loaded = require_success(ant.load(artifact, expected=artifact.expectation))
        loaded_report = loaded.inspect().to_dict()
        json.dumps(loaded_report, allow_nan=False)
        assert set(report) == set(loaded_report)
        assert loaded.answer == result.answer

        # 4. the retained study exists; refresh either re-reads the same rows or is
        #    refused by construction -- never an untyped crash.
        study = result.study
        if case.refresh == "same":
            updated = require_success(ant.estimate(ant.rebind(study, data=data)))
            assert type(updated) is type(result)
            assert updated.program_id == result.program_id
        elif case.refresh == "refused":
            with pytest.raises(
                CausalUnsupportedError, match="bound to the prepared evaluation rows"
            ):
                require_success(ant.estimate(ant.rebind(study, data=data)))


def test_policy_claim_does_not_crash() -> None:
    """Regression: result.claim() read a mistyped field name on every policy result."""
    result = require_success(ant.analyze({"y": _bernoulli(1)["y"]}, query=_policy_query()))
    assert "policy value" in result.claim().lower()
