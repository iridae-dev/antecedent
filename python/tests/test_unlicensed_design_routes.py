"""The unlicensed Bayesian and design routes through ``analyze`` against independent references.

The sequential panel is a fixed linear congruential stream whose numpy reference values are pinned
in the estimator's own tests. The policy and regime fixtures carry hand scores. The history match
is recomputed here by brute force with the Abadie-Imbens variance written from its per-unit
coefficients. Every route is unlicensed, and the matching posterior is a typed refusal.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import policy
from antecedent.errors import CausalError, CausalUnsupportedError
from antecedent.ids import Estimator
from antecedent.outcomes import require_success
from antecedent.quasi import PeriodHistoryMatch
from antecedent.regimes import LongitudinalRegime, SequentialContrast

MASK = (1 << 64) - 1
SUBJECTS = 80
Z95 = 1.959963984540054


def _panel() -> dict[str, np.ndarray]:
    state = 12345

    def uniform() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        return (state >> 11) / 9007199254740992.0

    cols: dict[str, list[float]] = {k: [] for k in ("l0", "a0", "l1", "a1", "y")}
    for _ in range(SUBJECTS):
        u1, u2, u3, u4, u5 = (uniform() for _ in range(5))
        l0 = 2.0 * u1 - 1.0
        a0 = 1.0 if u2 < 0.5 + 0.2 * l0 else 0.0
        l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u3 - 1.0) * 0.5
        a1 = 1.0 if u4 < 0.5 + 0.1 * (l1 - 0.4) else 0.0
        y = 1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u5 - 1.0)
        for key, value in zip(("l0", "a0", "l1", "a1", "y"), (l0, a0, l1, a1, y), strict=True):
            cols[key].append(value)
    return {key: np.asarray(values) for key, values in cols.items()}


def _contrast_query() -> SequentialContrast:
    return SequentialContrast(
        outcome="y",
        first_treatment="a0",
        second_treatment="a1",
        baseline=["l0"],
        intermediate=["l1"],
        active=(1, 1),
        control=(0, 0),
        folds=[int(row % 4 in (1, 2)) for row in range(SUBJECTS)],
        min_probability=0.05,
    )


def test_sequential_contrast_normal_interval_matches_the_pinned_reference() -> None:
    result = require_success(
        ant.analyze(
            _panel(), query=_contrast_query(), refute="none", execution=ant.Execution(seed=3)
        )
    )
    fit = result.sequential_contrast
    assert fit is not None
    assert abs(fit.active_mean - 3.5770033958969876) < 1e-4
    assert abs(fit.control_mean - 1.3461500907566415) < 1e-4
    assert abs(fit.contrast - 2.230853305140346) < 1e-4
    scores = np.asarray(fit.contrast_scores)
    se = scores.std(ddof=1) / np.sqrt(scores.size)
    assert fit.standard_error == pytest.approx(se, rel=1e-9)
    assert fit.interval is not None
    assert fit.interval[0] == pytest.approx(scores.mean() - Z95 * se, abs=1e-6)
    assert fit.licensed is False and fit.calibration_measured is False
    assert fit.uncertainty == "normal_interval_calibration_unmeasured"
    assert result.answer.detail.startswith("unlicensed_sequential_contrast")


def test_sequential_contrast_posterior_is_the_modular_dirichlet_posterior() -> None:
    result = require_success(
        ant.analyze(
            _panel(),
            query=_contrast_query(),
            inference=ant.Bayesian(backend="conjugate", n_draws=2000),
            estimator=Estimator.BAYESIAN_SEQUENTIAL_CROSSFIT_CONTRAST,
            refute="none",
            execution=ant.Execution(seed=3),
        )
    )
    fit = result.sequential_contrast
    assert fit is not None and fit.interval is None and fit.standard_error is None
    assert result.posterior is not None
    scores = np.asarray(fit.contrast_scores)
    expected = np.sqrt(scores.var() / (scores.size + 1.0))
    draws = _draws(result)[:, 0]
    assert draws.mean() == pytest.approx(fit.contrast, abs=0.05)
    assert draws.std(ddof=1) == pytest.approx(expected, rel=0.08)


def test_sequential_contrast_refusals_and_estimator_ids_are_typed() -> None:
    with pytest.raises(CausalUnsupportedError) as mismatch:
        require_success(
            ant.analyze(
                _panel(),
                query=_contrast_query(),
                estimator=Estimator.BAYESIAN_SEQUENTIAL_CROSSFIT_CONTRAST,
                refute="none",
            )
        )
    assert mismatch.value.reason_code == "estimator_inference_mismatch"
    floor = SequentialContrast(
        outcome="y",
        first_treatment="a0",
        second_treatment="a1",
        baseline=["l0"],
        intermediate=["l1"],
        active=(1, 1),
        control=(0, 0),
        min_probability=0.5,
    )
    with pytest.raises(CausalUnsupportedError) as positivity:
        require_success(ant.analyze(_panel(), query=floor, refute="none"))
    assert positivity.value.reason_code == "positivity_violated"


def _draws(result: object) -> np.ndarray:
    """The retained joint draws (stored quantity by quantity): a row per draw, a column per quantity."""
    artifact = ant.inference.decode_posterior_artifact(
        bytes(result.export(quantity="record"))  # type: ignore[attr-defined]
    )
    flat = np.asarray(artifact.draws)
    return flat.reshape(len(artifact.quantity_names), artifact.n_draws).T


def _policy_query(**overrides: object) -> policy.PolicyValue:
    args: dict[str, object] = {
        "outcome": "y",
        "assignment": [True, True, False, False],
        "propensity": 0.5,
        "policy": policy.BinaryPolicy([True] * 4),
        "reference": policy.BinaryPolicy([False] * 4),
        "evaluation_subject_ids": ["a", "b", "c", "d"],
    }
    args.update(overrides)
    return policy.PolicyValue(**args)  # type: ignore[arg-type]


def test_binary_policy_posterior_matches_the_hand_scores() -> None:
    result = require_success(
        ant.analyze(
            {"y": np.array([3.0, 5.0, 2.0, 1.0])},
            query=_policy_query(),
            inference=ant.Bayesian(backend="conjugate", n_draws=6000),
            refute="none",
            execution=ant.Execution(seed=5),
        )
    )
    value = result.policy_value
    assert value is not None
    assert value.policy_value == pytest.approx(4.0)
    assert value.reference_value == pytest.approx(1.5)
    assert value.incremental_value == pytest.approx(2.5)
    assert result.posterior is not None
    draws = _draws(result)
    scores = np.array([6.0, 10.0, -4.0, -2.0])
    assert draws[:, 0].mean() == pytest.approx(2.5, abs=0.1)
    assert draws[:, 0].std(ddof=1) == pytest.approx(np.sqrt(scores.var() / 5.0), rel=0.08)
    assert np.allclose(draws[:, 0], draws[:, 1] - draws[:, 2], atol=1e-9)


def _regime_query(**overrides: object) -> LongitudinalRegime:
    args: dict[str, object] = {
        "outcome": "y",
        "treatment_history": [[True, True], [True, False], [False, True], [False, False]],
        "actions": [True, True],
        "treatment_probabilities": [[0.5, 0.5]] * 4,
        "subject_ids": ["a", "b", "c", "d"],
        "method": "sequential_dr",
        "q_predictions": [[2.0, 3.0], [2.0, 3.0], [1.0, 2.0], [1.0, 2.0]],
        "observation_history": [[True, True]] * 4,
        "outcome_observed": [True] * 4,
        "fold_ids": [0, 1, 0, 1],
        "prediction_fold_ids": [0, 1, 0, 1],
        "excluded_fold_predictions": True,
        "minimum_probability": 0.1,
    }
    args.update(overrides)
    return LongitudinalRegime(**args)  # type: ignore[arg-type]


def test_sequential_regime_posterior_matches_the_hand_recursion() -> None:
    result = require_success(
        ant.analyze(
            {"y": np.array([5.0, 4.0, 3.0, 0.0])},
            query=_regime_query(),
            inference=ant.Bayesian(backend="conjugate", n_draws=6000),
            refute="none",
            execution=ant.Execution(seed=6),
        )
    )
    regime = result.longitudinal_regime
    assert regime is not None and regime.value == pytest.approx(4.5)
    assert regime.value_interval_95 is None
    assert result.posterior is not None
    draws = _draws(result)[:, 0]
    scores = np.array([12.0, 4.0, 1.0, 1.0])
    assert draws.mean() == pytest.approx(4.5, abs=0.15)
    assert draws.std(ddof=1) == pytest.approx(np.sqrt(scores.var() / 5.0), rel=0.08)


def test_sequential_regime_posterior_refuses_other_methods() -> None:
    ipw = _regime_query(
        method="ipw",
        q_predictions=None,
        observation_history=None,
        prediction_fold_ids=None,
        excluded_fold_predictions=False,
    )
    with pytest.raises(CausalError) as raised:
        require_success(
            ant.analyze(
                {"y": np.array([5.0, 4.0, 3.0, 0.0])},
                query=ipw,
                inference=ant.Bayesian(backend="conjugate", n_draws=100),
                refute="none",
            )
        )
    assert raised.value.reason_code == "route_not_supported"
    assert "Next attempt" in str(raised.value)


HISTORY = [
    # period, history, outcome, treated, weight
    (1, [1.0, 3.0], 10.0, 1.0, 1.0),
    (1, [1.0, 1.0], 4.0, 0.0, 1.0),
    (1, [3.0, 3.0], 6.0, 0.0, 1.0),
    (1, [2.8, 3.0], 9.0, 1.0, 2.0),
    (2, [2.0, 2.0], 5.0, 0.0, 1.0),
    (2, [6.0, 6.0], 7.0, 0.0, 1.0),
    (2, [5.0, 6.0], 12.0, 1.0, 1.5),
    (2, [9.0, 9.0], 20.0, 1.0, 1.0),
    (3, [1.0, 1.0], 3.0, 1.0, 1.0),
]


def _history_data() -> dict[str, np.ndarray]:
    return {
        "h1": np.array([row[1][0] for row in HISTORY]),
        "h2": np.array([row[1][1] for row in HISTORY]),
        "y": np.array([row[2] for row in HISTORY]),
        "treated": np.array([row[3] for row in HISTORY]),
        "period": np.array([float(row[0]) for row in HISTORY]),
        "weight": np.array([row[4] for row in HISTORY]),
    }


def _brute_force(caliper: float | None) -> tuple[float, float, list[int], list[int]]:
    score = [float(np.mean(row[1])) for row in HISTORY]
    queries: list[int] = []
    donors: list[int] = []
    for i, row in enumerate(HISTORY):
        if row[3] != 1.0:
            continue
        pool = [j for j, other in enumerate(HISTORY) if other[3] == 0.0 and other[0] == row[0]]
        if not pool:
            continue
        j = min(pool, key=lambda k: (abs(score[k] - score[i]), k))
        if caliper is None or abs(score[j] - score[i]) <= caliper:
            queries.append(i)
            donors.append(j)
    weight = np.array([HISTORY[i][4] for i in queries])
    diff = np.array([HISTORY[i][2] - HISTORY[j][2] for i, j in zip(queries, donors, strict=True)])
    share = weight / weight.sum()
    tau = float((share * diff).sum())
    donated = {d: float(share[[k for k, x in enumerate(donors) if x == d]].sum()) for d in donors}

    def noise(i: int) -> float | None:
        # Abadie-Imbens: half the squared difference to the nearest other unit of the same arm
        # and period (lowest input index on a tie); None when the unit has no such neighbour.
        peers = [
            j
            for j, other in enumerate(HISTORY)
            if j != i and other[3] == HISTORY[i][3] and other[0] == HISTORY[i][0]
        ]
        if not peers:
            return None
        j = min(peers, key=lambda k: (abs(score[k] - score[i]), k))
        return 0.5 * (HISTORY[i][2] - HISTORY[j][2]) ** 2

    variance = 0.0
    for k, i in enumerate(queries):
        v = noise(i)
        if v is None:
            return tau, float("nan"), queries, donors
        variance += float(share[k]) ** 2 * v
    for d, c in donated.items():
        v = noise(d)
        if v is None:
            return tau, float("nan"), queries, donors
        variance += c * c * v
    return tau, float(np.sqrt(variance)), queries, donors


@pytest.mark.parametrize("caliper", [2.0, None])
def test_period_history_match_matches_the_brute_force_reference(caliper: float | None) -> None:
    query = PeriodHistoryMatch(
        outcome="y",
        treated="treated",
        history=["h1", "h2"],
        period="period",
        weight="weight",
        caliper=caliper,
    )
    result = require_success(ant.analyze(_history_data(), query=query, refute="none"))
    fit = result.history_match
    assert fit is not None
    tau, se, queries, donors = _brute_force(caliper)
    assert list(fit.matched_queries) == queries
    assert list(fit.matched_donors) == donors
    assert fit.contrast == pytest.approx(tau, abs=1e-12)
    assert fit.standard_error == pytest.approx(se, abs=1e-12)
    assert fit.licensed is False
    assert fit.uncertainty == "abadie_imbens_standard_error_calibration_unmeasured_no_interval"


def test_bayesian_matching_is_the_typed_boundary() -> None:
    query = PeriodHistoryMatch(
        outcome="y", treated="treated", history=["h1", "h2"], period="period", weight="weight"
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        require_success(
            ant.analyze(
                _history_data(), query=query, inference=ant.Bayesian(n_draws=50), refute="none"
            )
        )
    assert raised.value.reason_code == "route_not_supported"
    message = str(raised.value)
    assert "no likelihood" in message and "matched within calendar periods" in message
    assert "next attempt: declare a Bayesian outcome model" in message
    rng = np.random.default_rng(3)
    z = rng.normal(size=300)
    t = (rng.uniform(size=300) < 1 / (1 + np.exp(-z))).astype(float)
    data = {"z": z, "treatment": t, "outcome": 1.5 * t + z + rng.normal(size=300)}
    graph = [("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")]
    for estimator in ("propensity.matching", "distance.matching"):
        with pytest.raises(CausalUnsupportedError) as static:
            require_success(
                ant.analyze(
                    data,
                    graph=graph,
                    query=ant.AverageEffect("treatment", "outcome"),
                    inference=ant.Bayesian(n_draws=32),
                    estimator=estimator,
                    refute="none",
                )
            )
        assert static.value.reason_code == "route_not_supported"
        assert "covariates or a propensity score" in str(static.value)


def _history_outcome_data(unequal_weights: bool = False) -> tuple[dict[str, np.ndarray], float]:
    """Three periods of 60 units; the effect ``1 + 2 * score`` grows with the history mean."""
    rows: dict[str, list[float]] = {k: [] for k in ("h1", "h2", "y", "treated", "period", "weight")}
    effects: list[float] = []
    for period in range(3):
        for k in range(60):
            base = float(np.sin(k * 0.37 + period))
            first, second = base, base + 0.1 * ((k % 5) - 2.0)
            score = (first + second) / 2.0
            treated = score + float(np.cos(k * 0.11)) * 0.8 > 0.0
            effect = 1.0 + 2.0 * score
            outcome = (
                1.5 * period + 0.5 * score + (effect if treated else 0.0) + 0.05 * np.sin(k * 1.3)
            )
            if treated:
                effects.append(effect)
            for key, value in zip(
                rows,
                (
                    first,
                    second,
                    outcome,
                    1.0 if treated else 0.0,
                    float(period),
                    2.0 if unequal_weights and k % 2 == 0 else 1.0,
                ),
                strict=True,
            ):
                rows[key].append(float(value))
    return {key: np.asarray(value) for key, value in rows.items()}, float(np.mean(effects))


def _history_outcome_query() -> PeriodHistoryMatch:
    return PeriodHistoryMatch(
        outcome="y", treated="treated", history=["h1", "h2"], period="period", weight="weight"
    )


def test_history_outcome_model_is_the_model_based_effect_on_the_treated() -> None:
    data, truth = _history_outcome_data()
    result = require_success(
        ant.analyze(
            data,
            query=_history_outcome_query(),
            inference=ant.Bayesian(backend="conjugate", n_draws=2000),
            estimator=Estimator.BAYESIAN_PERIOD_HISTORY_OUTCOME_MODEL,
            refute="none",
            execution=ant.Execution(seed=5),
        )
    )
    assert result.posterior is not None
    assert result.history_match is None, "a model-based effect is not a matched contrast"
    assert _draws(result)[:, 0].mean() == pytest.approx(truth, abs=0.06)


def test_history_outcome_model_refusals_are_typed_and_name_the_next_attempt() -> None:
    data, _ = _history_outcome_data()
    with pytest.raises(CausalUnsupportedError) as matching:
        require_success(
            ant.analyze(
                data,
                query=_history_outcome_query(),
                inference=ant.Bayesian(backend="conjugate", n_draws=50),
                refute="none",
            )
        )
    assert matching.value.reason_code == "route_not_supported"
    assert "bayesian.period_history_outcome_model" in str(matching.value)
    with pytest.raises(CausalUnsupportedError) as mismatch:
        require_success(
            ant.analyze(
                data,
                query=_history_outcome_query(),
                estimator=Estimator.BAYESIAN_PERIOD_HISTORY_OUTCOME_MODEL,
                refute="none",
            )
        )
    assert mismatch.value.reason_code == "estimator_inference_mismatch"
    weighted, _ = _history_outcome_data(unequal_weights=True)
    with pytest.raises(CausalError) as unequal:
        require_success(
            ant.analyze(
                weighted,
                query=_history_outcome_query(),
                inference=ant.Bayesian(backend="conjugate", n_draws=50),
                estimator=Estimator.BAYESIAN_PERIOD_HISTORY_OUTCOME_MODEL,
                refute="none",
            )
        )
    assert "equal sampling weights" in str(unequal.value)
    assert "Next attempt" in str(unequal.value)


def test_bayesian_gcomp_on_the_treated_is_a_model_based_effect_not_a_matching_posterior() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=1200)
    t = (rng.uniform(size=1200) < 1 / (1 + np.exp(-1.5 * z))).astype(float)
    y = t * (1.0 + 2.0 * z) + z + 0.3 * rng.normal(size=1200)
    data = {"z": z, "t": t, "y": y}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    truth = float(np.mean(1.0 + 2.0 * z[t > 0.5]))
    result = require_success(
        ant.analyze(
            data,
            graph=graph,
            query=ant.AverageEffect("t", "y", target_population=ant.population.Treated()),
            inference=ant.Bayesian(backend="conjugate", n_draws=2000),
            refute=False,
            execution=ant.Execution(seed=4),
        )
    )
    assert result.ate == pytest.approx(truth, abs=0.1)
    assert result.posterior is not None
