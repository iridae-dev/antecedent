"""Categorical-choice evidence-to-decision lifecycle (Python parity).

Mirrors ``choice_evidence::tests::lifecycle_runs_prior_to_ranking_and_refuses_orphan_stages``
in ``antecedent-design``; every number is computed by the Rust core.
"""

from __future__ import annotations

import pytest
from antecedent import choice_evidence as ce
from antecedent.choice_evidence import (
    Candidate,
    ChoiceEvidenceRefusal,
    Frame,
    Observations,
    Rule,
    Source,
)

from _refusal import assert_registered_refusal

TARGET = Frame(["a", "b", "c"], "panel")


def _source(**kwargs) -> Source:
    base = {
        "label": "wave1",
        "frame": TARGET,
        "counts": [20, 12, 8],
        "observation_ids": ["obs.wave1"],
        "discount": 0.5,
    }
    return Source(**{**base, **kwargs})


def test_lifecycle_from_prior_to_ranking() -> None:
    out = ce.evaluate(
        TARGET,
        [1, 1, 1],
        sources=[_source()],
        observations=Observations([14, 9, 7], ["obs.wave2"]),
        rule=Rule(0.95, 0.05, 10, 500, 0.05),
        values=[1.0, 1.0, 1.0],
        candidates=[Candidate("wave3", 40), Candidate("wave3_small", 10)],
    )
    assert out.prior["alpha"] == [11.0, 7.0, 5.0]
    assert out.prior["calibration"] == "unmeasured"
    inference = out.inference
    assert inference is not None
    assert abs(sum(inference["probability_best"]) - 1.0) < 1e-9
    assert inference["alpha"] == [25.0, 16.0, 12.0]
    assert out.look is not None and out.look["action"] in {"continue", "select"}
    ranking = out.ranking
    assert ranking is not None and ranking[0]["method"]["kind"] == "exact"
    assert ranking[0]["evsi"] >= ranking[1]["evsi"] >= 0.0


def test_refusals_are_typed_and_registered() -> None:
    other = Frame(["a", "b", "c"], "elsewhere")
    with pytest.raises(ChoiceEvidenceRefusal) as caught:
        ce.evaluate(TARGET, [1, 1, 1], sources=[_source(frame=other)])
    assert_registered_refusal(caught.value)
    assert caught.value.detail == "choice_evidence.population_mismatch"
    assert caught.value.remedy
    declared = _source(frame=other, transfer_rationale="same market, later wave")
    assert ce.evaluate(TARGET, [1, 1, 1], sources=[declared]).prior["sources"][0]["transferred"]
    with pytest.raises(ChoiceEvidenceRefusal) as caught:
        ce.evaluate(
            TARGET,
            [1, 1, 1],
            sources=[_source()],
            observations=Observations([1, 1, 1], ["obs.wave1"]),
        )
    assert caught.value.detail == "choice_evidence.overlapping_observations"
    with pytest.raises(ChoiceEvidenceRefusal) as caught:
        ce.evaluate(TARGET, [1, 1, 1], rule=Rule(0.95, 0.05, 10, 500))
    assert_registered_refusal(caught.value)
    behavioural = Frame(["a", "b", "c"], "panel", "observed_behavior")
    with pytest.raises(ChoiceEvidenceRefusal) as caught:
        ce.evaluate(TARGET, [1, 1, 1], sources=[_source(frame=behavioural)])
    assert caught.value.detail == "choice_evidence.meaning_mismatch"
