"""Multi-level observational treatment contrasts against a known simulated truth.

Three treatment levels with a covariate-dependent effect: ``E[Y(k) | z] = ALPHA[k] + BETA[k] z``
and a multinomial-logit assignment. The full-population effect of level 2 against level 0 is
``ALPHA[2] - ALPHA[0] = 3`` (``z`` has mean zero); the pairwise-subpopulation effect averages
``3 + z`` over the covariate density among rows at level 0 or 2, computed here by numerical
integration independent of the estimator. The two differ by about 0.4.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import multi_level_treatment
from antecedent.errors import CausalError

ALPHA = np.array([0.0, 1.0, 3.0])
BETA = np.array([1.0, 1.0, 2.0])
LOGIT_1 = (0.4, 0.8)
LOGIT_2 = (-0.3, -0.6)


def _draw(n: int = 8000, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    weights = np.stack(
        [np.ones(n), np.exp(LOGIT_1[0] + LOGIT_1[1] * z), np.exp(LOGIT_2[0] + LOGIT_2[1] * z)],
        axis=1,
    )
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    cumulative = probabilities.cumsum(axis=1)
    level = (rng.uniform(size=n)[:, None] > cumulative).sum(axis=1)
    y = ALPHA[level] + BETA[level] * z + rng.normal(size=n)
    return {"t": level.astype(float), "y": y, "z": z}


def _pairwise_truth() -> float:
    z = np.linspace(-9.0, 9.0, 18001)
    density = np.exp(-z * z / 2.0)
    weights = np.stack(
        [np.ones_like(z), np.exp(LOGIT_1[0] + LOGIT_1[1] * z), np.exp(LOGIT_2[0] + LOGIT_2[1] * z)]
    )
    at_pair = (weights[0] + weights[2]) / weights.sum(axis=0)
    effect = (ALPHA[2] - ALPHA[0]) + (BETA[2] - BETA[0]) * z
    return float((effect * at_pair * density).sum() / (at_pair * density).sum())


def test_both_targets_recover_their_own_truth_and_differ() -> None:
    data = _draw()
    common = {"active": 2.0, "control": 0.0, "adjust": {"z": data["z"]}}
    full = multi_level_treatment.multi_level_effect(
        data["y"], data["t"], target="full_population", levels=[0.0, 1.0, 2.0], **common
    )
    pair = multi_level_treatment.multi_level_effect(
        data["y"], data["t"], target="pairwise_subpopulation", **common
    )
    assert full.target_population == "all complete-case rows"
    assert pair.target_population == "rows at the two declared levels"
    assert full.excluded_rows == 0
    assert pair.excluded_rows == int((data["t"] == 1.0).sum())
    assert abs(full.effect - 3.0) < 4 * full.se_analytic
    assert abs(pair.effect - _pairwise_truth()) < 4 * pair.se_analytic
    assert full.effect - pair.effect > 0.1
    assert full.calibration == "unmeasured"
    for mean, alpha in zip(full.level_means, ALPHA, strict=True):
        assert abs(mean.mean - alpha) < 4 * mean.se


def test_a_row_at_an_undeclared_level_is_refused_not_dropped() -> None:
    data = _draw(1200, seed=5)
    with pytest.raises(CausalError) as raised:
        multi_level_treatment.multi_level_effect(
            data["y"],
            data["t"],
            active=2.0,
            control=0.0,
            target="full_population",
            levels=[0.0, 2.0],
            adjust={"z": data["z"]},
        )
    assert raised.value.reason_code == "treatment_level_undeclared"
    assert "treatment_levels.full_population_undeclared_level" in str(raised.value)


def test_the_target_has_no_default_and_full_population_needs_its_levels() -> None:
    data = _draw(300, seed=6)
    with pytest.raises(TypeError, match="target"):
        multi_level_treatment.multi_level_effect(  # type: ignore[call-arg]
            data["y"], data["t"], active=2.0, control=0.0
        )
    with pytest.raises(ValueError, match="declared levels"):
        multi_level_treatment.multi_level_effect(
            data["y"], data["t"], active=2.0, control=0.0, target="full_population"
        )
