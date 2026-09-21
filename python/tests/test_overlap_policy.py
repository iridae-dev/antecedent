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
    Aipw,
    DistanceMatching,
    Overlap,
    PropensityMatching,
    PropensityStratification,
    PropensityWeighting,
)

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
    return ant.analyze(
        _low_overlap(),
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y"),
        estimator=estimator,
        refute="none",
        bootstrap=0,
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
    assert trimmed.answer.value != default.answer.value
    assert t["inference_binding_id"] != d["inference_binding_id"]


def test_a_non_default_overlap_never_borrows_default_calibration() -> None:
    clipped = _frequentist(Aipw(bootstrap=0, overlap=Overlap(clip=0.05)))
    calibration = clipped.inspect().to_dict()["calibration"]
    assert calibration["status"] in ("unavailable", "scope_not_assessed")
    assert calibration["reason"] == "estimator_grid_not_measured"
    assert calibration["record_id"] is None


def test_overlap_config_dict_spelling_and_validation() -> None:
    dict_spelled = ant.analyze(
        _low_overlap(),
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y"),
        estimator="aipw",
        estimator_config={"bootstrap_replicates": 0, "overlap": {"clip": 0.05, "trim": None}},
        refute="none",
    )
    typed = _frequentist(Aipw(bootstrap=0, overlap=Overlap(clip=0.05)))
    assert dict_spelled.answer == typed.answer
    with pytest.raises(ValueError):
        Overlap(clip=0.5)
    with pytest.raises(ValueError):
        ant.analyze(
            _low_overlap(),
            graph=STATIC_DAG,
            query=ant.AverageEffect("t", "y"),
            estimator="linear.adjustment.ate",
            estimator_config={"overlap": {"clip": 0.05}},
        )


def test_default_overlap_is_the_native_default_policy():
    from antecedent._defaults import OMITTED

    assert Overlap().clip == OMITTED["overlap_clip"] == 0.01
    assert Overlap().trim == OMITTED["overlap_trim"] is None
