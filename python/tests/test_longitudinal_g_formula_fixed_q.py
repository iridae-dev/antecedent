"""Retained g-formula intervals require a caller-declared fixed known reward law."""

from __future__ import annotations

import antecedent as ant
import pytest
from antecedent.regimes import LongitudinalRegime


def _query(subjects: int, *, known: bool) -> LongitudinalRegime:
    baseline = [(-1.0 if subject % 2 == 0 else 1.0) for subject in range(subjects)]
    return LongitudinalRegime(
        outcome="y",
        method="g_formula",
        treatment_history=[[False, False]] * subjects,
        actions=[True, True],
        treatment_probabilities=[[0.5, 0.5]] * subjects,
        subject_ids=[f"subject-{subject}" for subject in range(subjects)],
        period_outcome_predictions=[[1.0 + 0.2 * x, 2.0 + 0.3 * x] for x in baseline],
        known_fixed_outcome_predictions=known,
    )


def test_fixed_known_q_interval_round_trips_and_default_stays_point_only() -> None:
    data = {"y": [0.0] * 400}
    result = ant.analyze(data, query=_query(400, known=True))
    value = result.longitudinal_regime
    assert value.value == pytest.approx(3.0)
    assert value.value_standard_error > 0
    assert value.value_interval_95[0] < 3.0 < value.value_interval_95[1]
    assert value.uncertainty == "pointwise_subject_score_conditional_fixed_known_q_95"
    assert "fixed_known_outcome_predictions" in " ".join(result.assumptions or [])
    artifact = ant.load(result.export()).artifact.payload
    assert artifact["query"]["longitudinal_regime"]["known_fixed_outcome_predictions"] is True
    assert artifact["longitudinal_regime"]["value_interval_95"] == pytest.approx(
        value.value_interval_95
    )

    default_result = ant.analyze(data, query=_query(400, known=False)).longitudinal_regime
    assert default_result.value_interval_95 is None
    assert default_result.interval_reason == "prediction_model_uncertainty_not_accounted"


def test_fixed_known_q_interval_withholds_thin_subjects() -> None:
    result = ant.analyze({"y": [0.0] * 50}, query=_query(50, known=True)).longitudinal_regime
    assert result.value_interval_95 is None
    assert "insufficient" in result.interval_reason
