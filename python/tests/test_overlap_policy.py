"""``overlap=Overlap(clip=..., trim=...)`` on the propensity-score estimators.

The clip and trim reach the Rust propensity estimators, change the answer on a
low-overlap design and the inference binding, and a non-default policy never
borrows calibration measured at the default.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.estimators import (
    DML,
    Aipw,
    DistanceMatching,
    DRLearner,
    Overlap,
    PropensityMatching,
    PropensityStratification,
    PropensityWeighting,
)
from antecedent.outcomes import require_success

STATIC_DAG = [("z", "t"), ("z", "y"), ("t", "y")]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------------------------------------------------- overlap


def _low_overlap(n: int = 1500, seed: int = 6) -> dict[str, np.ndarray]:
    """Strong confounding pushes many propensities past 0.01 / 0.99."""
    rng = np.random.default_rng(seed)
    z = 1.5 * rng.normal(size=n)
    t = (rng.uniform(size=n) < _sigmoid(1.6 * z)).astype(float)
    y = t * (1.0 + z) + z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _frequentist(estimator: Any) -> Any:
    return require_success(
        ant.analyze(
            _low_overlap(),
            graph=STATIC_DAG,
            query=ant.AverageEffect("t", "y"),
            estimator=estimator,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
        )
    )


@pytest.mark.parametrize(
    "estimator",
    [Aipw, PropensityWeighting, PropensityMatching, PropensityStratification, DistanceMatching],
)
def test_overlap_reaches_rust_changes_the_answer_and_the_binding(estimator: Any) -> None:
    default = _frequentist(estimator(bootstrap=0))
    spelled = _frequentist(estimator(bootstrap=0, overlap=Overlap()))
    trimmed = _frequentist(estimator(bootstrap=0, overlap=Overlap(clip=0.01, trim=0.1)))
    d, s, t = (r.inspect().to_dict() for r in (default, spelled, trimmed))
    # The spelled-out default is the default construction.
    assert spelled.answer == default.answer
    assert s["inference_binding_id"] == d["inference_binding_id"]
    assert t["inference_binding_id"] != d["inference_binding_id"]
    # The effect is 1 + z with mean 1. Heavy tail weights bias the default (observed 1.19 to
    # 1.47); trimming moves each estimator by 0.08 to 0.38 and brings it within 0.21 of 1.
    assert abs(trimmed.answer.value - default.answer.value) > 0.05
    assert abs(trimmed.answer.value - 1.0) < 0.25
    assert abs(trimmed.answer.value - 1.0) < abs(default.answer.value - 1.0)


def test_a_non_default_overlap_never_borrows_default_calibration() -> None:
    clipped = _frequentist(Aipw(bootstrap=0, overlap=Overlap(clip=0.05)))
    calibration = clipped.inspect().to_dict()["calibration"]
    assert calibration["status"] in ("unavailable", "scope_not_assessed")
    assert calibration["reason"] == "estimator_grid_not_measured"
    assert calibration["record_id"] is None


def test_overlap_config_dict_spelling_and_validation() -> None:
    dict_spelled = require_success(
        ant.analyze(
            _low_overlap(),
            graph=STATIC_DAG,
            query=ant.AverageEffect("t", "y"),
            estimator="aipw",
            estimator_config={"bootstrap_replicates": 0, "overlap": {"clip": 0.05, "trim": None}},
            refute="none",
        )
    )
    typed = _frequentist(Aipw(bootstrap=0, overlap=Overlap(clip=0.05)))
    assert dict_spelled.answer == typed.answer
    with pytest.raises(ValueError, match="Overlap.clip must lie in"):
        Overlap(clip=0.5)
    with pytest.raises(ValueError, match="is not valid for estimator"):
        require_success(
            ant.analyze(
                _low_overlap(),
                graph=STATIC_DAG,
                query=ant.AverageEffect("t", "y"),
                estimator="linear.adjustment.ate",
                estimator_config={"overlap": {"clip": 0.05}},
            )
        )


def test_default_overlap_is_the_native_default_policy():
    from antecedent._defaults import OMITTED

    assert Overlap().clip == OMITTED["overlap_clip"] == 0.01
    assert OMITTED["overlap_trim"] is None
    assert Overlap().trim is None


def _saturated(n: int = 2000, seed: int = 11) -> dict[str, np.ndarray]:
    """A strong, converged propensity model whose tail scores reach within 1e-8 of 0 or 1."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < _sigmoid(9.0 * z)).astype(float)
    y = t + z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _saturated_outcome(estimator: Any) -> Any:
    return ant.analyze(
        _saturated(),
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y"),
        estimator=estimator,
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
    )


def test_saturated_propensities_without_an_overlap_policy_refuse_as_positivity():
    outcome = _saturated_outcome(PropensityWeighting(bootstrap=0, overlap=Overlap(clip=None)))
    refusal = outcome.refusal
    assert refusal is not None
    assert refusal.code == "positivity_violated"
    assert "within 1e-8 of 0 or 1" in refusal.message
    assert "overlap trim" in refusal.remedy
    fields = refusal.fields["refusal_fields"]
    assert fields["reason"] == "boundary_saturated" and fields["boundary_count"] > 0


@pytest.mark.parametrize(
    "overlap", [Overlap(clip=None, trim=0.01), Overlap(clip=0.01), Overlap()], ids=str
)
@pytest.mark.parametrize("estimator", [Aipw, PropensityWeighting])
def test_a_declared_overlap_policy_governs_saturated_propensities(
    estimator: Any, overlap: Overlap
) -> None:
    result = require_success(_saturated_outcome(estimator(bootstrap=0, overlap=overlap)))
    assert np.isfinite(result.answer.value)
    # The effect is exactly 1. Observed: AIPW 0.92 (trim) and 1.07 (clip); weighting 0.97
    # (trim) and 1.95 (clip 0.01 caps weights at 100, so its tail bias is large but bounded).
    bound = 0.2 if estimator is Aipw or overlap.trim is not None else 1.1
    assert abs(result.answer.value - 1.0) < bound, result.answer.value


@pytest.mark.parametrize("estimator", [DML, DRLearner])
def test_cross_fitted_learners_follow_the_same_overlap_rule(estimator: Any) -> None:
    refused = _saturated_outcome(estimator())
    assert refused.refusal is not None
    assert refused.refusal.code == "positivity_violated"
    governed = require_success(_saturated_outcome(estimator(overlap=Overlap(trim=0.01))))
    assert np.isfinite(governed.answer.value)
    assert abs(governed.answer.value - 1.0) < 0.2, governed.answer.value  # observed 0.9075
