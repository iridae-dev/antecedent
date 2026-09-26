"""Evidence for caller-supplied sequential g-formula value predictions."""

import numpy as np
import pytest
from antecedent.regimes import evaluate_sequential_gformula


def test_static_sequential_gformula_averages_period_rewards_and_preserves_folds():
    result = evaluate_sequential_gformula(
        period_outcome_predictions=[[1.0, 2.0], [3.0, 4.0]],
        treatment_history=[[0, 0], [1, 1]],
        regime=[False, True],
        treatment_probabilities=[[0.8, 0.8], [0.9, 0.9]],
        censoring_survival=[[1.0, 0.8], [0.9, 1.0]],
        subject_ids=["s1", "s2"],
        fold_ids=[0, 1],
    )

    assert result.value == pytest.approx(5.0)
    assert result.subject_count == 2
    assert result.minimum_regime_action_probability == pytest.approx(0.1)
    assert result.minimum_censoring_survival == pytest.approx(0.8)
    assert result.fold_ownership == (("s1", 0), ("s2", 1))
    assert result.uncertainty == "not_estimated"
    assert result.support_status == "caller_supplied_conditional_outcome_predictions"
    assert result.crossfit_status == "not_claimed"
    assert any("conditional reward predictions" in item for item in result.assumptions)


def test_dynamic_sequential_gformula_uses_observed_histories_for_policy_actions():
    visited = []

    def regime(time, past_treatments, covariates):
        visited.append((time, past_treatments))
        return covariates[-1][0] > 0

    result = evaluate_sequential_gformula(
        period_outcome_predictions=[[2.0, 3.0]],
        treatment_history=[[False, True]],
        regime=regime,
        treatment_probabilities=[[0.5, 0.5]],
        covariate_history=[[[1.0], [-1.0]]],
        subject_ids=["unit-7"],
        fold_ids=[4],
    )

    assert visited == [(0, ()), (1, (False,))]
    assert result.value == pytest.approx(5.0)
    assert result.fold_ownership == (("unit-7", 4),)


def test_gformula_refuses_positivity_censoring_and_fold_ownership_violations():
    base = dict(
        period_outcome_predictions=[[1.0, 2.0], [3.0, 4.0]],
        treatment_history=[[0, 0], [1, 1]],
        regime=[False, True],
        treatment_probabilities=[[0.8, 0.8], [0.9, 0.9]],
        subject_ids=["s1", "s2"],
        fold_ids=[0, 1],
    )
    with pytest.raises(ValueError, match="treatment positivity"):
        evaluate_sequential_gformula(
            **{**base, "treatment_probabilities": [[0.005, 0.8], [0.9, 0.9]]}
        )
    with pytest.raises(ValueError, match="censoring positivity"):
        evaluate_sequential_gformula(**base, censoring_survival=[[1.0, 0.0], [0.9, 1.0]])
    with pytest.raises(ValueError, match="one row per unique subject"):
        evaluate_sequential_gformula(**{**base, "subject_ids": ["s1", "s1"]})
    with pytest.raises(ValueError, match="one non-negative signed 64-bit integer per subject"):
        evaluate_sequential_gformula(**{**base, "fold_ids": [0, 1.5]})
    with pytest.raises(ValueError, match="non-negative signed 64-bit integer"):
        evaluate_sequential_gformula(**{**base, "fold_ids": [0, -1]})


def test_gformula_rejects_nonfinite_or_misaligned_conditional_predictions():
    base = dict(
        period_outcome_predictions=[[1.0, np.inf]],
        treatment_history=[[0, 0]],
        regime=[False, True],
        treatment_probabilities=[[0.5, 0.5]],
        subject_ids=["s1"],
        fold_ids=[0],
    )
    with pytest.raises(ValueError, match="finite"):
        evaluate_sequential_gformula(**base)
    with pytest.raises(ValueError, match="match period_outcome_predictions"):
        evaluate_sequential_gformula(
            **{**base, "period_outcome_predictions": [[1.0, 2.0]], "treatment_history": [[0]]}
        )
