"""Temporal designs as typed queries of ``analyze`` / ``prepare``.

The query dataclasses name columns of one unit-period table. The result is
``result.temporal_design``: it must equal the standalone
:func:`antecedent.temporal_designs.estimate_temporal_design` result on the same data, carry a
structured unlicensed answer, and read back equal from ``load(result.export())``. Regime
survival reports its whole-procedure bootstrap standard error (frequentist) or posterior
standard deviation (Bayesian). Coverage is not measured here.
"""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.temporal_designs import (
    DoseExcursion,
    FactorialExcursion,
    InterferenceExcursion,
    MultiArmExcursion,
    OutcomeMisclassification,
    RegimeSurvivalContrast,
    SmoothedDoseContrast,
    TemporalDesignEstimate,
    TwoArmExcursion,
    estimate_temporal_design,
)


def _two_arm_table(units: int = 160, periods: int = 6, seed: int = 3) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    table: dict[str, list] = {"y": [], "unit": [], "period": [], "a": [], "p": []}
    for i in range(units):
        level = rng.uniform(-0.5, 0.5)
        previous_y, previous_a = 0.0, 0.0
        for t in range(periods):
            p = 0.65 if previous_y > 0 else 0.35
            a = float(rng.uniform() < p)
            y = level + 1.2 * previous_a + rng.uniform(-0.5, 0.5)
            for key, value in zip(table, (y, f"u{i}", t, a, p), strict=True):
                table[key].append(value)
            previous_y, previous_a = y, a
    return table


def _two_arm_query() -> TwoArmExcursion:
    return TwoArmExcursion("y", "unit", "period", assignment="a", probability="p", lag=1)


def _same(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(left, right, strict=True)
    )


@pytest.mark.parametrize("bayesian", [False, True])
def test_analyze_equals_the_standalone_module(bayesian: bool) -> None:
    table = _two_arm_table()
    inference = antecedent.Bayesian(n_draws=200) if bayesian else None
    result = antecedent.analyze(table, query=_two_arm_query(), inference=inference, seed=5)
    design = result.temporal_design
    assert isinstance(design, TemporalDesignEstimate)
    standalone = estimate_temporal_design(
        {"y": table["y"]},
        outcome="y",
        units=table["unit"],
        periods=table["period"],
        design={
            "design": "two_arm",
            "assignment": [value == 1.0 for value in table["a"]],
            "probability": table["p"],
        },
        lag=1,
        inference="bayesian" if bayesian else "frequentist",
        n_draws=200 if bayesian else None,
        seed=5,
    )
    assert design.coordinates == standalone.coordinates
    assert _same(design.estimate, standalone.estimate)
    assert _same(design.standard_error, standalone.standard_error)
    assert _same(design.lower, standalone.lower)
    assert _same(design.upper, standalone.upper)
    assert not design.licensed and design.calibration == "unmeasured"
    assert (design.draws is not None) is bayesian


def test_the_answer_is_structured_and_reads_back_equal() -> None:
    result = antecedent.analyze(_two_arm_table(), query=_two_arm_query())
    answer = result.answer
    assert answer.kind == "structured"
    assert answer.detail is not None and answer.detail.startswith("unlicensed_temporal_design_")
    structured = answer.structured
    assert structured is not None
    assert structured["design"] == "two_arm"
    assert structured["licensed"] is False and structured["calibration"] == "unmeasured"
    assert structured["estimate"][0] == result.temporal_design.estimate[0]
    assert "two_arm temporal design" in result.claim()
    artifact = result.export()
    loaded = antecedent.load(artifact)
    assert loaded.export() == artifact
    assert loaded.temporal_design == result.temporal_design
    assert loaded.answer == result.answer


def test_prepare_and_estimate_on_the_same_table_reproduce_analyze() -> None:
    table = _two_arm_table()
    direct = antecedent.analyze(table, query=_two_arm_query(), seed=7)
    prepared = antecedent.prepare(table, query=_two_arm_query(), seed=7)
    clicked = prepared.estimate(table)
    assert clicked.temporal_design == direct.temporal_design


def _regime_table(subjects: int = 900, horizon: int = 3, seed: int = 11) -> dict[str, list]:
    rng = np.random.default_rng(seed)

    def expit(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    table: dict[str, list] = {
        "event": [],
        "unit": [],
        "period": [],
        "treated": [],
        "censored": [],
        "l": [],
    }
    for i in range(subjects):
        current = rng.uniform(-1.0, 1.0)
        previous = 0.0
        for k in range(horizon):
            if k > 0:
                current = 0.5 * current - 0.4 * previous + rng.uniform(-0.5, 0.5)
            a = float(rng.uniform() < expit(0.8 * current))
            c = float(rng.uniform() < 0.03)
            y = float(c == 0.0 and rng.uniform() < expit(-2.5 + 0.8 * current - 0.7 * a))
            for key, value in zip(table, (y, f"s{i}", k, a, c, current), strict=True):
                table[key].append(value)
            previous = a
            if c or y:
                break
    return table


def _regime_query() -> RegimeSurvivalContrast:
    return RegimeSurvivalContrast(
        "event",
        "unit",
        "period",
        treatment="treated",
        censored="censored",
        confounders=("l",),
        horizon=3,
        min_probability=0.02,
    )


def test_regime_survival_reports_its_bootstrap_standard_error() -> None:
    result = antecedent.analyze(_regime_table(), query=_regime_query(), bootstrap=99)
    design = result.temporal_design
    assert design.design == "regime_survival"
    assert design.replicates is not None and design.replicates[0] >= 90
    assert design.bootstrap_standard_error == design.standard_error
    assert all(value > 0.0 for value in design.bootstrap_standard_error)
    assert design.posterior_standard_deviation is None
    assert "whole-procedure bootstrap standard error" in result.claim()
    assert antecedent.load(result.export()).temporal_design == design


def test_bayesian_regime_survival_reports_its_posterior_standard_deviation() -> None:
    result = antecedent.analyze(
        _regime_table(), query=_regime_query(), inference=antecedent.Bayesian(n_draws=200)
    )
    design = result.temporal_design
    assert design.draws == 200
    assert design.bootstrap_standard_error is None
    assert design.posterior_standard_deviation == design.standard_error
    assert "posterior standard deviation" in result.claim()


def test_every_design_query_compiles_to_its_design() -> None:
    rng = np.random.default_rng(13)
    rows = 6 * 120
    table = {
        "y": rng.normal(size=rows).tolist(),
        "unit": [f"u{i // 6}" for i in range(rows)],
        "period": [i % 6 for i in range(rows)],
        "arm": rng.choice(["a", "b", "c"], size=rows).tolist(),
        "pa": [1 / 3] * rows,
        "pb": [1 / 3] * rows,
        "pc": [1 / 3] * rows,
        "f1": (rng.uniform(size=rows) < 0.5).astype(float).tolist(),
        "f2": (rng.uniform(size=rows) < 0.5).astype(float).tolist(),
        "half": [0.5] * rows,
        "cluster": [f"c{(i // 6) // 4}" for i in range(rows)],
        "dose": rng.uniform(0.0, 2.0, size=rows).tolist(),
        "density": [0.5] * rows,
    }
    queries = {
        "multi_arm": MultiArmExcursion(
            "y",
            "unit",
            "period",
            assignment="arm",
            probabilities=("pa", "pb", "pc"),
            arms=("a", "b", "c"),
        ),
        "factorial": FactorialExcursion(
            "y",
            "unit",
            "period",
            first="f1",
            first_probability="half",
            second="f2",
            second_probability="half",
        ),
        "interference": InterferenceExcursion(
            "y", "unit", "period", assignment="f1", probability="half", cluster="cluster"
        ),
        "dose": DoseExcursion(
            "y",
            "unit",
            "period",
            dose="dose",
            density="density",
            bandwidth=0.4,
            target=SmoothedDoseContrast(active=1.5, control=0.5),
        ),
    }
    for name, query in queries.items():
        design = antecedent.analyze(table, query=query).temporal_design
        assert design is not None and design.design == name, name
        assert not design.licensed


def test_query_options_validate_and_misclassification_needs_a_binary_outcome() -> None:
    with pytest.raises(CausalValueError):
        TwoArmExcursion("y", "unit", "period", assignment="a", probability="p", lag=-1)
    with pytest.raises(CausalValueError):
        OutcomeMisclassification(1, -1, 1, 1)
    query = TwoArmExcursion(
        "y",
        "unit",
        "period",
        assignment="a",
        probability="p",
        misclassification=OutcomeMisclassification(90, 10, 85, 15),
    )
    with pytest.raises(CausalError):
        antecedent.analyze(_two_arm_table(units=40), query=query)


def test_a_graph_or_refutation_is_not_applicable() -> None:
    with pytest.raises(CausalUnsupportedError):
        antecedent.analyze(_two_arm_table(units=40), query=_two_arm_query(), refute="cheap")
