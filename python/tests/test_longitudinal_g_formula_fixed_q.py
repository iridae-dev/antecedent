"""Retained g-formula intervals require a caller-declared fixed known reward law."""

from __future__ import annotations

import antecedent as ant
import pytest
from antecedent import composition_bundle as cb
from antecedent.outcomes import require_success
from antecedent.regimes import FittedRegimeQ, LongitudinalRegime


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
    result = require_success(ant.analyze(data, query=_query(400, known=True)))
    value = result.longitudinal_regime
    assert value.value == pytest.approx(3.0)
    assert value.value_standard_error > 0
    assert value.value_interval_95[0] < 3.0 < value.value_interval_95[1]
    assert value.uncertainty == "pointwise_subject_score_conditional_fixed_known_q_95"
    assert "fixed_known_outcome_predictions" in " ".join(result.assumptions or [])
    artifact = require_success(ant.load(bytes(result.export()))).decoded_artifact.payload
    assert artifact["query"]["longitudinal_regime"]["known_fixed_outcome_predictions"] is True
    assert artifact["longitudinal_regime"]["value_interval_95"] == pytest.approx(
        value.value_interval_95
    )

    default_result = require_success(
        ant.analyze(data, query=_query(400, known=False))
    ).longitudinal_regime
    assert default_result.value_interval_95 is None
    assert default_result.interval_reason == "prediction_model_uncertainty_not_accounted"


def test_fixed_known_q_interval_withholds_thin_subjects() -> None:
    result = require_success(
        ant.analyze({"y": [0.0] * 50}, query=_query(50, known=True))
    ).longitudinal_regime
    assert result.value_interval_95 is None
    assert "insufficient" in result.interval_reason


@pytest.mark.parametrize("external", [False, True])
def test_fitted_two_period_q_bootstraps_declared_subject_partitions(external: bool) -> None:
    n = 800
    original_index = [i // 16 * 16 + (i % 2) * 8 + i // 2 % 8 for i in range(n)]
    treatment = [[bool(i & 1), bool(i & 2)] for i in original_index]
    baseline = [float(i // 4 % 2) for i in original_index]
    folds = [i % 2 for i in range(n)]
    outcomes = [
        1.0 + row[0] + 2.0 * row[1] + 3.0 * x + 2.0 * (i // 16 % 11 - 5)
        for i, row, x in zip(original_index, treatment, baseline, strict=True)
    ]
    data = {"outcome": outcomes, "baseline": baseline}
    query = LongitudinalRegime(
        outcome="outcome",
        method="fitted_g_formula",
        fitted_q=FittedRegimeQ(
            "baseline", bootstrap_replicates=199, external_training_fold=0 if external else None
        ),
        treatment_history=treatment,
        actions=[True, True],
        treatment_probabilities=[[0.5] * 2] * n,
        subject_ids=[f"fitted-{i}" for i in range(n)],
        fold_ids=folds,
    )
    predictions = []
    for i in range(n):
        if external and folds[i] == 0:
            continue
        training = [
            outcomes[j]
            for j in range(n)
            if folds[j] != folds[i] and all(treatment[j]) and baseline[j] == baseline[i]
        ]
        predictions.append(sum(training) / len(training))
    result = require_success(ant.analyze(data, query=query))
    value = result.longitudinal_regime
    assert value.value == pytest.approx(sum(predictions) / len(predictions))
    assert value.effective_sample_size == len(predictions)
    assert value.fitted_q_bootstrap_replicates_ok == 199
    assert value.uncertainty == "pointwise_whole_subject_refit_percentile_95"
    assert result.calibration.functional is None
    assert result.calibration.status == "not_measured"
    sourced = bytes(result.export(source_data=data))
    artifact = require_success(ant.load(sourced)).replay()
    assert artifact.acceptance.replayed
    assert artifact.decoded_artifact.payload["longitudinal_regime"][
        "value_interval_95"
    ] == pytest.approx(value.value_interval_95)
    bundle = cb.BundleBuilder().add_artifact(sourced, node_id="regime").build()
    consumed = require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    )
    assert consumed.value("regime", "upper_95") == value.value_interval_95[1]

    revised_data = {
        "outcome": [y + 1000.0 * fold for y, fold in zip(outcomes, folds, strict=True)],
        "baseline": baseline,
    }
    revised = require_success(ant.analyze(revised_data, query=query)).longitudinal_regime
    if external:
        assert revised.value == value.value
        assert revised.value_interval_95 == value.value_interval_95
    else:
        assert revised.value > value.value + 400.0

    changed_data = {
        "outcome": [y + 5.0 * (fold == 0) for y, fold in zip(outcomes, folds, strict=True)],
        "baseline": baseline,
    }
    changed = require_success(ant.analyze(changed_data, query=query))
    changed_bundle = (
        cb.BundleBuilder()
        .add_artifact(
            bytes(changed.export(source_data=changed_data)),
            node_id="regime",
        )
        .build()
    )
    assert changed_bundle.identity != bundle.identity
    changed_bytes = bytes(changed_bundle.export())
    with pytest.raises(cb.ExpectedIdentityMismatchRefusal):
        require_success(cb.consume_bundle(changed_bytes, expected_identity=bundle.identity))
    verified = require_success(
        cb.consume_bundle(changed_bytes, expected_identity=changed_bundle.identity)
    )
    assert verified.value("regime", "regime_value") > value.value
    assert verified.value("regime", "upper_95") == changed.longitudinal_regime.value_interval_95[1]
