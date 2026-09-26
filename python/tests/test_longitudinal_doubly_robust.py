"""Known-truth and refusal evidence for sequential DR regime evaluation."""

import numpy as np
import pytest
from antecedent.regimes import evaluate_sequential_doubly_robust


def _base():
    return dict(
        outcomes=[7.0, 99.0, np.nan],
        outcome_observed=[True, True, False],
        observation_history=[[True, True], [True, True], [True, False]],
        treatment_history=[[False, False], [False, True], [False, False]],
        regime=[False, False],
        q_predictions=[[1.0, 3.0], [5.0, 4.0], [2.0, 6.0]],
        treatment_probabilities=np.full((3, 2), 0.5),
        censoring_probabilities=np.full((3, 2), 0.8),
        subject_ids=["s1", "s2", "s3"],
        fold_ids=[0, 1, 0],
        prediction_fold_ids=[0, 1, 0],
    )


def test_sequential_dr_backward_recursion_known_truth_and_fold_ownership():
    result = evaluate_sequential_doubly_robust(**_base())

    # Backward scores are 31 for s1, 2.5 for the t=1 treatment mismatch s2,
    # and 12 for s3 censored after t=0; their subject mean is 45.5 / 3.
    assert result.value == pytest.approx(45.5 / 3)
    assert result.subject_count == 3
    assert result.minimum_regime_action_probability == pytest.approx(0.5)
    assert result.minimum_censoring_probability == pytest.approx(0.8)
    assert result.fold_ownership == (("s1", 0), ("s2", 1), ("s3", 0))
    assert result.uncertainty == "not_estimated"
    assert result.support_status == "caller_supplied_sequential_Q_and_probabilities"
    assert result.crossfit_status == "fold_ids_aligned_but_crossfit_not_independently_verified"
    assert any("cross-fitted by subject fold" in item for item in result.assumptions)


def test_sequential_dr_accepts_dynamic_regime_from_history():
    def policy(time, past_treatments, covariates):
        return covariates[-1][0] > 0.0

    result = evaluate_sequential_doubly_robust(
        outcomes=[5.0],
        outcome_observed=[True],
        observation_history=[[True, True]],
        treatment_history=[[True, False]],
        regime=policy,
        q_predictions=[[2.0, 3.0]],
        treatment_probabilities=[[0.5, 0.5]],
        covariate_history=[[[1.0], [-1.0]]],
        subject_ids=["unit-1"],
        fold_ids=[0],
        prediction_fold_ids=[0],
    )
    # Both observed actions match dynamic actions [True, False].
    assert result.value == pytest.approx(12.0)


def test_sequential_dr_refuses_positivity_and_observation_failures():
    base = _base()
    with pytest.raises(ValueError, match="treatment positivity"):
        evaluate_sequential_doubly_robust(
            **{**base, "treatment_probabilities": [[0.005, 0.5], [0.5, 0.5], [0.5, 0.5]]}
        )
    with pytest.raises(ValueError, match="censoring positivity"):
        evaluate_sequential_doubly_robust(
            **{**base, "censoring_probabilities": [[0.8, 0.005], [0.8, 0.8], [0.8, 0.8]]}
        )
    with pytest.raises(ValueError, match="monotone after censoring"):
        evaluate_sequential_doubly_robust(
            **{**base, "observation_history": [[True, True], [True, True], [False, True]]}
        )


def test_sequential_dr_checks_prediction_fold_alignment_and_unique_subject_rows():
    base = _base()
    with pytest.raises(ValueError, match="fold ownership must match"):
        evaluate_sequential_doubly_robust(
            **{**base, "prediction_fold_ids": [1, 1, 0]}
        )
    with pytest.raises(ValueError, match="one row per unique subject"):
        evaluate_sequential_doubly_robust(
            **{**base, "subject_ids": ["s1", "s1", "s3"]}
        )
