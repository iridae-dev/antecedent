"""The multi-level difference-in-differences contrast against an exact rational reference.

``generate_categorical_did_reference.py`` forms every figure in ``Fraction`` arithmetic from the
definitions (cell means, change-score means, the cluster score sandwich with its G/(G-1)
factor, the odds-weighted augmented comparison). The cases are unbalanced groups with a two-subject
level, tied changes (a level of zero variance), a 1e8 outcome offset and string levels with a
non-smallest reference.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.quasi import (
    AugmentedPanelDiD,
    DifferenceInDifferences,
    PanelDifferenceInDifferences,
    estimate_categorical_did,
    estimate_did,
)

from _repo_text import load_json

CASES = load_json(Path(__file__).parent / "fixtures" / "categorical_did_reference.json")["cases"]


def exact(text: str) -> Fraction:
    numerator, denominator = text.split("/")
    return Fraction(int(numerator), int(denominator))


def close(got: float, want: Fraction, rel: float, what: str) -> None:
    want = float(want)
    assert abs(got - want) <= rel * max(abs(want), 1e-300) + 1e-12, f"{what}: got {got} want {want}"


def long_panel(case):
    ids, y, group, after = [], [], [], []
    for index, (pre, post, level) in enumerate(case["subjects"]):
        for value, period in ((pre, False), (post, True)):
            ids.append(f"s{index}")
            y.append(float(value))
            group.append(level)
            after.append(period)
    return {"y": np.array(y), "id": ids, "group": group, "after": after}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_the_two_by_two_contrast_of_every_level_equals_the_exact_cell_mean_reference(case):
    data = long_panel(case)
    fit = estimate_categorical_did(
        data, DifferenceInDifferences("y", "group", "after"), reference=case["reference"]
    )
    assert fit.reference == case["reference"]
    assert {entry.level for entry in fit.contrasts} == {
        entry["level"] for entry in case["contrasts"].values()
    }
    for entry in fit.contrasts:
        want = case["contrasts"][str(entry.level)]
        close(entry.estimate, exact(want["did_2x2"]), 1e-12 if case["name"] != "offset_four_string_levels" else 1e-7, f"{case['name']} {entry.level}")
        assert entry.standard_error is None


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_the_panel_contrast_and_cluster_standard_error_equal_the_exact_reference(case):
    data = long_panel(case)
    fit = estimate_categorical_did(
        data,
        PanelDifferenceInDifferences("y", "id", "group", "after"),
        reference=case["reference"],
    )
    assert fit.design == "categorical_balanced_two_period_panel"
    for entry in fit.contrasts:
        want = case["contrasts"][str(entry.level)]
        rel = 1e-7 if case["name"] == "offset_four_string_levels" else 1e-9
        close(entry.estimate, exact(want["panel"]), rel, f"{case['name']} {entry.level} estimate")
        variance = float(exact(want["panel_variance"]))
        close(entry.standard_error, Fraction(math.sqrt(variance)), 1e-6 if rel > 1e-8 else 1e-9, f"{case['name']} {entry.level} se")
        assert entry.level_units == want["level_subjects"]
        assert entry.reference_units == want["reference_subjects"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_the_augmented_panel_contrast_equals_the_exact_odds_weighted_reference(case):
    n = len(case["subjects"])
    data = {
        "pre": np.array([float(row[0]) for row in case["subjects"]]),
        "post": np.array([float(row[1]) for row in case["subjects"]]),
        "id": [f"s{i}" for i in range(n)],
        "group": [row[2] for row in case["subjects"]],
        "p": np.array(case["propensity"]),
        "m": np.array(case["prediction"]),
    }
    query = AugmentedPanelDiD("pre", "post", "id", "group", "p", "m", True)
    levels = [entry["level"] for entry in case["contrasts"].values()]
    fit = estimate_categorical_did(
        data, query, reference=case["reference"], nuisance={level: ("p", "m") for level in levels}
    )
    for entry in fit.contrasts:
        want = case["contrasts"][str(entry.level)]
        rel = 1e-6 if case["name"] == "offset_four_string_levels" else 1e-9
        close(entry.estimate, exact(want["augmented"]), rel, f"{case['name']} {entry.level}")


def test_a_binary_treated_column_still_matches_the_binary_kernel_bit_for_bit():
    case = CASES[0]
    data = long_panel(case)
    keep = [g in (0, 1) for g in data["group"]]
    subset = {
        "y": data["y"][keep],
        "id": [v for v, k in zip(data["id"], keep) if k],
        "group": [g == 1 for g, k in zip(data["group"], keep) if k],
        "after": [v for v, k in zip(data["after"], keep) if k],
    }
    query = DifferenceInDifferences("y", "group", "after")
    binary = estimate_did(subset, query).estimate
    categorical = estimate_categorical_did(
        {**subset, "group": [int(g) for g in subset["group"]]}, query, reference=0
    )
    assert categorical.contrast(1).estimate == binary
    panel = require_success(
        ant.analyze(subset, query=PanelDifferenceInDifferences("y", "id", "group", "after"))
    ).panel_did
    categorical_panel = estimate_categorical_did(
        {**subset, "group": [int(g) for g in subset["group"]]},
        PanelDifferenceInDifferences("y", "id", "group", "after"),
        reference=0,
    )
    assert categorical_panel.contrast(1).estimate == panel.estimate
    assert categorical_panel.contrast(1).standard_error == panel.standard_error


def test_the_binary_forms_still_refuse_a_third_level_and_name_the_categorical_contrast():
    data = long_panel(CASES[0])
    with pytest.raises(CausalValueError, match="Next attempt: estimate_categorical_did") as caught:
        estimate_did(data, DifferenceInDifferences("y", "group", "after"))
    assert caught.value.reason_code == "treatment_level_undeclared"
    with pytest.raises(CausalError, match="Next attempt: estimate_categorical_did"):
        require_success(
            ant.analyze(data, query=PanelDifferenceInDifferences("y", "id", "group", "after"))
        )


def test_the_categorical_contrast_refuses_what_it_cannot_read():
    data = long_panel(CASES[0])
    with pytest.raises(CausalValueError) as caught:
        estimate_categorical_did(
            data, PanelDifferenceInDifferences("y", "id", "group", "after"), reference=9
        )
    assert caught.value.reason_code == "treatment_level_undeclared"
    one_level = {**data, "group": [0] * len(data["group"])}
    with pytest.raises(CausalValueError) as caught:
        estimate_categorical_did(
            one_level, PanelDifferenceInDifferences("y", "id", "group", "after")
        )
    assert caught.value.reason_code == "arm_not_populated"
    n = len(CASES[0]["subjects"])
    augmented = {
        "pre": np.zeros(n),
        "post": np.ones(n),
        "id": [f"s{i}" for i in range(n)],
        "group": [row[2] for row in CASES[0]["subjects"]],
        "p": np.full(n, 0.5),
        "m": np.zeros(n),
    }
    with pytest.raises(Exception) as caught:
        estimate_categorical_did(
            augmented, AugmentedPanelDiD("pre", "post", "id", "group", "p", "m", True)
        )
    assert caught.value.reason_code == "required_option_missing"
