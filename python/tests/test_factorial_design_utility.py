from __future__ import annotations

import numpy as np
import pytest
from antecedent import factorial


def _balanced_fixture() -> tuple[dict[str, np.ndarray], list[bool], list[bool]]:
    a = [False, False, False, False, True, True, True, True]
    b = [False, False, True, True, False, False, True, True]
    outcome = np.array([1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 10.0, 10.0])
    return {"y": outcome}, a, b


def test_factorial_2x2_recovers_known_cell_means_and_contrasts() -> None:
    data, a, b = _balanced_fixture()
    result = factorial.estimate(
        data,
        factor_a=a,
        factor_b=b,
        design=factorial.FactorialDesign(0.5, 0.5),
        outcome="y",
    )

    assert result.cell_means == pytest.approx({"00": 1.0, "01": 4.0, "10": 3.0, "11": 10.0})
    assert result.cell_support == {"00": 2, "01": 2, "10": 2, "11": 2}
    assert result.factor_a_effect == pytest.approx(4.0)
    assert result.factor_b_effect == pytest.approx(5.0)
    assert result.interaction_effect == pytest.approx(4.0)
    assert result.factor_a_variance_bound > 0.0
    assert result.factor_b_variance_bound > 0.0
    assert result.interaction_variance_bound > 0.0
    assert "independently Bernoulli randomized" in result.assumptions[0]
    assert "no confidence interval" in result.uncertainty_semantics


def test_factorial_2x2_refuses_probability_positivity_failure() -> None:
    data, a, b = _balanced_fixture()
    with pytest.raises(ValueError, match="strictly between zero and one"):
        factorial.estimate(
            data,
            factor_a=a,
            factor_b=b,
            design=factorial.FactorialDesign(0.0, 0.5),
            outcome="y",
        )


def test_factorial_2x2_refuses_unobserved_cell_and_bad_assignment_alignment() -> None:
    data, a, b = _balanced_fixture()
    with pytest.raises(ValueError, match="observed support"):
        factorial.estimate(
            data,
            factor_a=a,
            factor_b=[False] * len(b),
            design=factorial.FactorialDesign(0.5, 0.5),
            outcome="y",
        )
    with pytest.raises(ValueError, match="one value per data row"):
        factorial.estimate(
            data,
            factor_a=a[:-1],
            factor_b=b,
            design=factorial.FactorialDesign(0.5, 0.5),
            outcome="y",
        )
