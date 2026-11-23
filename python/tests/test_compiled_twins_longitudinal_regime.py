"""The direct regime evaluators warn, and the compiled ``LongitudinalRegime`` query matches."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import regimes
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success

N = 60
PERIODS = 2


def _fixture():
    treatment, probability, censoring, rewards, q = [], [], [], [], []
    outcome = []
    for i in range(N):
        y = 1.0 + 0.01 * (i % 17)
        row_a, row_p, row_g, row_r, row_q = [], [], [], [], []
        for t in range(PERIODS):
            treated = (i * (t + 3) + t * 5 + i // 4) % 5 < 3
            row_a.append(treated)
            row_p.append(0.4 + 0.02 * ((i + 2 * t) % 6))
            row_g.append(1.0 - 0.03 * ((3 * i + t) % 5))
            row_r.append(0.5 + 0.2 * t + 0.005 * (i % 13))
            row_q.append(0.3 + 0.1 * t + 0.01 * (i % 11))
            y += (0.5 + 0.3 * t) if treated else 0.0
        treatment.append(row_a)
        probability.append(row_p)
        censoring.append(row_g)
        rewards.append(row_r)
        q.append(row_q)
        outcome.append(y)
    return {
        "outcome": np.array(outcome),
        "treatment": np.array(treatment),
        "probability": np.array(probability),
        "censoring": np.array(censoring),
        "rewards": np.array(rewards),
        "q": np.array(q),
        "ids": [f"s{i}" for i in range(N)],
        "folds": [i % 2 for i in range(N)],
    }


def _compiled(fx, **fields):
    query = regimes.LongitudinalRegime(
        outcome="y",
        treatment_history=fx["treatment"].tolist(),
        actions=[True] * PERIODS,
        treatment_probabilities=fx["probability"].tolist(),
        censoring_probabilities=fx["censoring"].tolist(),
        subject_ids=fx["ids"],
        fold_ids=fx["folds"],
        **fields,
    )
    return require_success(
        ant.analyze({"y": fx["outcome"].tolist()}, query=query, refute="none")
    ).longitudinal_regime


def test_ipw_direct_path_warns_and_the_compiled_regime_matches_bit_for_bit():
    fx = _fixture()
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_regime_value"):
        direct = regimes.evaluate_regime_value(
            fx["outcome"],
            fx["treatment"],
            [True] * PERIODS,
            fx["probability"],
            censoring_survival=fx["censoring"],
        )
    compiled = _compiled(fx, method="ipw")
    assert compiled.value == direct.value
    assert compiled.effective_sample_size == direct.effective_sample_size
    assert compiled.matched_observed_fraction == direct.matched_observed_fraction
    assert compiled.maximum_weight == direct.maximum_weight


def test_g_formula_direct_path_warns_and_the_compiled_regime_matches_bit_for_bit():
    fx = _fixture()
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_sequential_gformula"):
        direct = regimes.evaluate_sequential_gformula(
            fx["rewards"],
            fx["treatment"],
            [True] * PERIODS,
            fx["probability"],
            subject_ids=fx["ids"],
            fold_ids=fx["folds"],
            censoring_survival=fx["censoring"],
        )
    compiled = _compiled(fx, method="g_formula", period_outcome_predictions=fx["rewards"].tolist())
    assert compiled.value == direct.value
    assert compiled.minimum_action_probability == direct.minimum_regime_action_probability
    assert compiled.minimum_censoring_probability == direct.minimum_censoring_survival


def test_sequential_dr_direct_path_warns_and_the_compiled_regime_matches_bit_for_bit():
    fx = _fixture()
    observed = np.ones((N, PERIODS), dtype=bool)
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_sequential_doubly_robust"):
        direct = regimes.evaluate_sequential_doubly_robust(
            fx["outcome"],
            observed[:, -1],
            observed,
            fx["treatment"],
            [True] * PERIODS,
            fx["q"],
            fx["probability"],
            subject_ids=fx["ids"],
            fold_ids=fx["folds"],
            prediction_fold_ids=fx["folds"],
            censoring_probabilities=fx["censoring"],
        )
    compiled = _compiled(
        fx,
        method="sequential_dr",
        q_predictions=fx["q"].tolist(),
        observation_history=observed.tolist(),
        prediction_fold_ids=fx["folds"],
        excluded_fold_predictions=True,
    )
    assert compiled.value == direct.value
    assert compiled.minimum_action_probability == direct.minimum_regime_action_probability
    assert compiled.minimum_censoring_probability == direct.minimum_censoring_probability
