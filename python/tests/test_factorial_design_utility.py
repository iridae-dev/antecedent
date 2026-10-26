from __future__ import annotations

import numpy as np
import pytest
from antecedent import factorial
from antecedent.outcomes import require_success


def _balanced_fixture() -> tuple[dict[str, np.ndarray], list[bool], list[bool]]:
    a = [False, False, False, False, True, True, True, True]
    b = [False, False, True, True, False, False, True, True]
    outcome = np.array([1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 10.0, 10.0])
    return {"y": outcome}, a, b


def test_factorial_2x2_recovers_known_cell_means_and_contrasts() -> None:
    data, a, b = _balanced_fixture()
    result = require_success(
        factorial.estimate(
            data,
            factor_a=a,
            factor_b=b,
            design=factorial.FactorialDesign(0.5, 0.5),
            outcome="y",
        )
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
        require_success(
            factorial.estimate(
                data,
                factor_a=a,
                factor_b=b,
                design=factorial.FactorialDesign(0.0, 0.5),
                outcome="y",
            )
        )


def test_factorial_2x2_refuses_unobserved_cell_and_bad_assignment_alignment() -> None:
    data, a, b = _balanced_fixture()
    with pytest.raises(ValueError, match="observed support"):
        require_success(
            factorial.estimate(
                data,
                factor_a=a,
                factor_b=[False] * len(b),
                design=factorial.FactorialDesign(0.5, 0.5),
                outcome="y",
            )
        )
    with pytest.raises(ValueError, match="one value per data row"):
        require_success(
            factorial.estimate(
                data,
                factor_a=a[:-1],
                factor_b=b,
                design=factorial.FactorialDesign(0.5, 0.5),
                outcome="y",
            )
        )


def test_bayesian_factorial_main_effect_matches_the_exact_beta_moments() -> None:
    import antecedent as ant
    from antecedent import experiment
    from antecedent.design_models import BernoulliArms

    # Cells in the declared order (A=0,B=0), (A=1,B=0), (A=0,B=1), (A=1,B=1), ten units each, with
    # 3, 6, 5 and 9 successes: Beta(1 + s, 11 - s) posteriors under a uniform prior.
    successes = [3, 6, 5, 9]
    primary = [bool(c % 2) for c in range(4) for _ in range(10)]
    second = [c >= 2 for c in range(4) for _ in range(10)]
    outcome = np.array([float(r < successes[c]) for c in range(4) for r in range(10)])
    randomization = experiment.FactorialRandomization(second, (10, 10, 10, 10), ("off", "on"))
    query = experiment.RandomizedEffect(
        "outcome",
        experiment.ExperimentDesign(
            randomization,
            primary,
            [f"account-{i}" for i in range(40)],
            [f"row-{i}" for i in range(40)],
        ),
    )
    result = require_success(
        ant.analyze(
            {"outcome": outcome},
            query=query,
            inference=ant.Bayesian(
                backend="conjugate", n_draws=4000, design_model=BernoulliArms(1.0, 1.0)
            ),
            estimator=ant.Estimator.BAYESIAN_RANDOMIZED_SUPERPOPULATION,
            seed=2,
            refute="none",
        )
    )
    theta = [(1 + s) / 12 for s in successes]  # order 00, 10, 01, 11
    variance = [(1 + s) * (11 - s) / (144 * 13) for s in successes]
    main_effect_a = 0.5 * (theta[1] + theta[3] - theta[0] - theta[2])
    sd_a = 0.5 * sum(variance) ** 0.5
    assert result.estimate.estimator_id == "bayesian.randomized_superpopulation"
    assert result.posterior.effect_mean == pytest.approx(main_effect_a, abs=5e-3)
    assert result.posterior.effect_sd == pytest.approx(sd_a, abs=6e-3)


def test_factorial_2x2_runs_through_the_compiled_randomized_route() -> None:
    data, a, b = _balanced_fixture()
    probability_a = [0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]
    result = require_success(
        factorial.estimate(
            data,
            factor_a=a,
            factor_b=b,
            design=factorial.FactorialDesign(probability_a, 0.5),
            outcome="y",
        )
    )
    y = data["y"]
    expected = {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}
    for i, outcome in enumerate(y):
        pa = probability_a[i] if a[i] else 1.0 - probability_a[i]
        cell = f"{int(a[i])}{int(b[i])}"
        expected[cell] += outcome / (pa * 0.5) / len(y)
    assert result.cell_means == pytest.approx(expected)
    assert result.interaction_effect == pytest.approx(
        expected["11"] - expected["10"] - expected["01"] + expected["00"]
    )
    assert result.interaction_variance_bound > 0.0
