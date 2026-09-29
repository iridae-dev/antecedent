from __future__ import annotations

import numpy as np
import pytest
from antecedent import analyze
from antecedent.errors import CausalCompileError, CausalValueError
from antecedent.estimation import PreparedAnalysis
from antecedent.quasi import (
    AugmentedPanelDiD,
    SyntheticDifferenceInDifferences,
)


def _synthetic_did_panel():
    units = ["treated", "d0", "d1", "d2"]
    unit_effect = {"treated": 10.0, "d0": 2.0, "d1": 10.0, "d2": 18.0}
    common = {1: 1.0, 2: 3.0, 3: -2.0, 4: 5.0}
    rows = [(unit, period) for unit in units for period in (1, 2, 3, 4)]
    return {
        "unit": [unit for unit, _ in rows],
        "period": [period for _, period in rows],
        "y": [
            unit_effect[unit] + common[period] + (7.0 if unit == "treated" and period == 4 else 0.0)
            for unit, period in rows
        ],
    }


def test_synthetic_did_recovers_known_effect_with_additive_unit_and_time_effects():
    result = analyze(
        _synthetic_did_panel(),
        query=SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 4),
    ).synthetic_did
    assert result.estimate == pytest.approx(7.0)
    assert result.n_donors == 3
    assert result.n_pre_periods == 3
    assert result.n_post_periods == 1
    assert sum(weight for _, weight in result.donor_weights) == pytest.approx(1.0)
    assert sum(weight for _, weight in result.time_weights) == pytest.approx(1.0)
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert "no_concurrent_treated_unit_specific_shock" in result.assumptions


def test_retained_synthetic_did_matches_direct_and_preserves_point_only_weights():
    rows = _synthetic_did_panel()
    query = SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 4)
    direct = analyze(rows, query=query).synthetic_did
    result = analyze(rows, query=query)
    fit = result.synthetic_did
    assert fit is not None
    assert fit.estimate == pytest.approx(direct.estimate)
    assert dict(fit.donor_weights) == pytest.approx(dict(direct.donor_weights))
    assert dict(fit.time_weights) == pytest.approx(dict(direct.time_weights))
    assert fit.support_status == "unlicensed_point_utility"
    assert fit.uncertainty == "point_only"
    assert np.isnan(result.estimate.se_analytic)
    prepared = PreparedAnalysis.prepare(rows, query=query)
    assert prepared.estimate(rows).synthetic_did == fit


def test_synthetic_did_exact_uniform_unit_assignment_uses_retained_analysis():
    import antecedent

    rows = _synthetic_did_panel()
    query = SyntheticDifferenceInDifferences(
        "y",
        "unit",
        "period",
        "treated",
        4,
        uniform_unit_randomization=True,
    )
    direct = analyze(rows, query=query).synthetic_did
    prepared = antecedent.prepare(rows, query=query)
    result = prepared.estimate()
    fit = result.synthetic_did
    assert fit is not None
    assert fit == direct
    assert len(fit.randomization_statistics) == 4
    observed = dict(fit.randomization_statistics)["treated"]
    assert fit.randomization_p_value == pytest.approx(
        sum(stat >= observed for _, stat in fit.randomization_statistics) / 4,
    )
    assert fit.uncertainty == "point_only_with_exact_unit_randomization_p_value_no_interval"
    assert "uniform_single_treated_unit_assignment" in fit.assumptions
    assert np.isnan(result.estimate.se_analytic)
    artifact = result.export()
    assert antecedent.load(artifact).export() == artifact
    with pytest.raises(CausalValueError, match="uniform_unit_randomization"):
        SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 4, 1)


def test_synthetic_did_refuses_unbalanced_and_insufficient_pre_support():
    query = SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 4)
    rows = _synthetic_did_panel()
    with pytest.raises(CausalCompileError, match="balanced panel"):
        analyze({key: value[:-1] for key, value in rows.items()}, query=query)
    with pytest.raises(CausalCompileError, match="two pre-periods"):
        analyze(
            rows,
            query=SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 2),
        )


def test_augmented_panel_did_recovers_known_effect_and_reports_overlap():
    subjects = [f"s{i}" for i in range(6)]
    treated = [True, True, True, False, False, False]
    baseline = np.array([10.0, 12.0, 13.0, 9.0, 15.0, 11.0])
    changes = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])
    result = analyze(
        {
            "id": subjects,
            "pre": baseline,
            "post": baseline + changes,
            "treated": treated,
            "p": np.full(6, 0.5),
            "m0": np.full(6, 2.0),
        },
        query=AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0", True),
    ).panel_did
    assert result.estimate == pytest.approx(3.0)
    assert result.treated_subjects == 3
    assert result.control_subjects == 3
    assert result.propensity_min == pytest.approx(0.5)
    assert result.propensity_max == pytest.approx(0.5)
    assert result.effective_control_sample_size == pytest.approx(3.0)
    assert result.nuisance_predictions_cross_fitted
    assert result.uncertainty == "point_only_no_standard_error"
    assert "strict_propensity_overlap" in result.assumptions


def test_augmented_panel_did_refuses_nonoverlap_and_duplicate_subjects():
    query = AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0")
    data = {
        "id": ["a", "b", "c", "d"],
        "pre": [0.0] * 4,
        "post": [1.0, 1.0, 0.0, 0.0],
        "treated": [True, True, False, False],
        "p": [0.5, 0.5, 0.0, 0.5],
        "m0": [0.0] * 4,
    }
    with pytest.raises(CausalCompileError, match="strictly between zero and one"):
        analyze(data, query=query)
    with pytest.raises(CausalValueError, match="unique row per subject"):
        analyze({**data, "id": ["a", "a", "c", "d"]}, query=query)


def test_augmented_panel_did_uses_retained_prepare_analyze_and_artifact():
    import antecedent

    query = AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0", True)
    data = {
        "id": [f"s{i}" for i in range(6)],
        "pre": [10.0, 12.0, 13.0, 9.0, 15.0, 11.0],
        "post": [15.0, 17.0, 18.0, 11.0, 17.0, 13.0],
        "treated": [True, True, True, False, False, False],
        "p": [0.5] * 6,
        "m0": [2.0] * 6,
    }
    direct = antecedent.analyze(data, query=query).panel_did
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.panel_did.estimate == pytest.approx(3.0)
    assert result.panel_did.effective_control_sample_size == pytest.approx(3.0)
    assert result.panel_did.uncertainty == "point_only_no_standard_error"
    assert result.panel_did.estimate == direct.estimate
    assert antecedent.analyze(data, query=query).panel_did == result.panel_did
    assert (
        prepared.estimate({**data, "post": [x + 1 for x in data["post"]]}).panel_did
        == result.panel_did
    )
    artifact = result.export()
    loaded = antecedent.load(artifact)
    assert loaded.export() == artifact
    section = antecedent.artifacts.loads(artifact).payload["panel_did"]
    assert "standard_error" not in section
    assert section["uncertainty"] == "point_only_no_standard_error"
    with pytest.raises(CausalCompileError, match="overlap|strictly between"):
        antecedent.analyze({**data, "p": [0.0] + [0.5] * 5}, query=query)
