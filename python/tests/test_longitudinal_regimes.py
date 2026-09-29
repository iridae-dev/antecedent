"""End-to-end contracts for point evaluation of longitudinal regimes."""

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.regimes import LongitudinalRegime, evaluate_regime_value


def test_known_truth_static_regime_value_uses_complete_treatment_histories():
    # Under sequential randomization with p=.5 at each of two times, the
    # treatment path (0, 0) has probability .25. Its sole observed outcome is 2.
    result = evaluate_regime_value(
        outcomes=[2.0, 0.0, 0.0, 0.0],
        treatment_history=[[0, 0], [0, 1], [1, 0], [1, 1]],
        regime=[False, False],
        treatment_probabilities=np.full((4, 2), 0.5),
    )
    assert result.value == pytest.approx(2.0)
    assert result.effective_sample_size == pytest.approx(1.0)
    assert result.matched_observed_fraction == pytest.approx(0.25)
    assert result.maximum_weight == pytest.approx(4.0)
    assert result.uncertainty == "not_estimated"
    assert result.support_status == "caller_supplied_sequential_probabilities"
    assert "sequential exchangeability" in result.assumptions[1]
    assert "sequential positivity" in result.assumptions[2]


def test_dynamic_regime_sees_only_observed_past_treatment_and_covariates():
    seen = []

    def regime(time, past_treatments, covariates):
        seen.append((time, past_treatments, covariates))
        return covariates[-1][0] > 0.0

    result = evaluate_regime_value(
        outcomes=[1.0],
        treatment_history=[[1, 0]],
        regime=regime,
        covariate_history=[[[1.0], [-1.0]]],
        treatment_probabilities=[[0.5, 0.5]],
    )
    assert [entry[0] for entry in seen] == [0, 1]
    assert seen[0][1] == ()
    assert seen[1][1] == (True,)
    assert seen[1][2] == ((1.0,), (-1.0,))
    assert result.value == pytest.approx(4.0)


def test_censoring_weights_apply_only_to_observed_matching_regime_trajectories():
    result = evaluate_regime_value(
        outcomes=[3.0, 100.0],
        treatment_history=[[0], [1]],
        regime=[False],
        treatment_probabilities=[[0.5], [0.5]],
        outcome_observed=[True, False],
        censoring_survival=[[0.5], [0.5]],
    )
    assert result.value == pytest.approx(6.0)
    assert result.matched_observed_fraction == pytest.approx(0.5)
    assert result.maximum_weight == pytest.approx(4.0)


def test_rejects_sequential_positivity_violation():
    with pytest.raises(ValueError, match="treatment positivity"):
        evaluate_regime_value(
            outcomes=[1.0],
            treatment_history=[[0]],
            regime=[False],
            treatment_probabilities=[[0.001]],
        )


def test_rejects_invalid_censoring_probability_and_dynamic_regime_output():
    with pytest.raises(ValueError, match="censoring positivity"):
        evaluate_regime_value(
            outcomes=[1.0],
            treatment_history=[[0]],
            regime=[False],
            treatment_probabilities=[[0.5]],
            censoring_survival=[[0.0]],
        )
    with pytest.raises(ValueError, match="must return a bool"):
        evaluate_regime_value(
            outcomes=[1.0],
            treatment_history=[[0]],
            regime=lambda *_: 1,
            covariate_history=[[[0.0]]],
            treatment_probabilities=[[0.5]],
        )


def test_refuses_regime_without_any_observed_matching_trajectory():
    with pytest.raises(ValueError, match="no observed trajectories"):
        evaluate_regime_value(
            outcomes=[1.0],
            treatment_history=[[1]],
            regime=[False],
            treatment_probabilities=[[0.5]],
        )


def test_stabilized_msm_recovers_known_additive_period_effects_and_clustered_se():
    histories = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 4, axis=0)
    residuals = np.tile([-1.0, 0.0, 0.0, 1.0], 4)
    y = 10.0 + 2.0 * histories[:, 0] + 3.0 * histories[:, 1] + residuals
    query = LongitudinalRegime.marginal_structural_model(
        outcome="y",
        treatment_history=histories,
        treatment_probabilities=np.full((16, 2), 0.5),
        stabilizing_numerator_probabilities=[0.5, 0.5],
        subject_ids=[f"s{i}" for i in range(16)],
        fold_ids=[i % 4 for i in range(16)],
    )
    result = antecedent.analyze({"y": y}, query=query).longitudinal_regime
    assert result.value == pytest.approx(10.0)
    assert result.period_effects == pytest.approx((2.0, 3.0))
    assert len(result.standard_errors) == 2
    assert all(value > 0.0 and np.isfinite(value) for value in result.standard_errors)
    assert result.effective_sample_size == pytest.approx(16.0)
    assert result.maximum_weight == pytest.approx(1.0)
    assert result.observed_subjects == 16
    assert result.method == "marginal_structural_model"
    assert result.uncertainty == "pointwise_subject_clustered_cr1_no_interval"
    assert result.support_status == "unlicensed_point_utility"


def test_msm_direct_accepts_strided_numpy_inputs():
    histories = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 4, axis=0)
    y = 10.0 + 2.0 * histories[:, 0] + 3.0 * histories[:, 1]
    padded_y = np.column_stack((y, y))
    padded_p = np.stack(
        (np.full_like(histories, 0.5, dtype=float), np.full_like(histories, 0.5, dtype=float)),
        axis=2,
    )
    query = LongitudinalRegime.marginal_structural_model(
        outcome="y",
        treatment_history=histories,
        treatment_probabilities=padded_p[:, :, 0],
        stabilizing_numerator_probabilities=np.array([0.5, 9.0, 0.5])[::2],
        outcome_observed=np.ones(32, dtype=bool)[::2],
        censoring_probabilities=padded_p[:, :, 0] * 0.0 + 1.0,
        subject_ids=[f"s{i}" for i in range(16)],
    )
    result = antecedent.analyze({"y": padded_y[:, 0]}, query=query).longitudinal_regime
    assert result.period_effects == pytest.approx((2.0, 3.0))


def test_msm_refuses_bad_weights_rank_deficiency_and_duplicate_subjects():
    y = np.arange(8.0)
    histories = np.column_stack((np.arange(8) % 2, np.arange(8) % 2))
    probabilities = np.full((8, 2), 0.5)

    def msm_query(treatment_probabilities, *, treatment_history=histories, subject_ids=None):
        return LongitudinalRegime.marginal_structural_model(
            outcome="y",
            treatment_history=treatment_history,
            treatment_probabilities=treatment_probabilities,
            stabilizing_numerator_probabilities=[0.5, 0.5],
            subject_ids=subject_ids or [f"s{i}" for i in range(8)],
        )

    with pytest.raises(CausalValueError, match="positivity fails"):
        antecedent.analyze({"y": y}, query=msm_query(np.full((8, 2), 0.001)))
    with pytest.raises(CausalUnsupportedError, match="rank-deficient"):
        antecedent.analyze({"y": y}, query=msm_query(probabilities))
    with pytest.raises(CausalValueError, match="distinct"):
        antecedent.analyze(
            {"y": y},
            query=msm_query(
                probabilities,
                treatment_history=np.column_stack((np.arange(8) % 2, (np.arange(8) // 2) % 2)),
                subject_ids=["duplicate"] * 8,
            ),
        )
