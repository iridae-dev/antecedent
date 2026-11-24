"""The direct sensitivity ranges of supplied evidence warn, and their compiled programs reproduce them.

``validation.matched_design_sensitivity`` (matched pairs, pair vectors and matched sets) and
``validation.supplied_evalue``, ``supplied_evalue_vector`` and ``supplied_evalue_posterior``
(estimates computed outside the library) compile a program of their own family and call the
same kernel as the deprecated direct functions on the same inputs, so every number and the
range (target and identity included) are the direct call's. The posterior ranges of a marginal
structural model run on the study's own compiled result.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import regimes
from antecedent import validation as v
from antecedent._deprecation import AntecedentDeprecationWarning

PAIRS = [1.5, -0.5, 2.0, 2.0, 0.0, 3.1, -1.5, 0.7, 4.2, 2.6, -2.6, 0.9, 1.1]
SETS = [([5.0, 1.0, 2.0], 0), ([3.0, 4.0, 0.0, 1.5], 1), ([2.0, 2.5], 1), ([6.0, 0.5, 1.0], 0)]


def _same(compiled, direct):
    summary = dict(compiled.summary)
    assert len(summary.pop("program")) == 64
    assert summary == direct.summary
    assert compiled.range == direct.range
    assert compiled.method == direct.method


def test_the_matched_design_entries_warn_and_the_compiled_program_reproduces_them():
    compiled = v.matched_design_sensitivity(PAIRS, gammas=[1.0, 1.5, 2.0])
    with pytest.warns(AntecedentDeprecationWarning, match="rosenbaum_bounds"):
        direct = v.rosenbaum_bounds(PAIRS, gammas=[1.0, 1.5, 2.0])
    _same(compiled, direct)
    # The bound at Gamma = 1 is the exact randomization p-value: 153 of the 4096 equally
    # likely sign vectors of the 12 non-zero pairs reach the observed rank sum 62.
    assert compiled.summary["statistic"] == 62.0
    assert compiled.summary["grid"][0]["p_value"] == pytest.approx(153 / 4096, rel=1e-12)

    columns = [PAIRS, [1.0, 2.0, 3.0, -0.25, 4.0, 5.0, 6.0, 0.5]]
    compiled = v.matched_design_sensitivity(columns, labels=["young", "old"])
    with pytest.warns(AntecedentDeprecationWarning, match="rosenbaum_bounds_vector"):
        direct = v.rosenbaum_bounds_vector(columns, ["young", "old"])
    _same(compiled, direct)

    for score in ("aligned_ranks", "outcomes"):
        compiled = v.matched_design_sensitivity(sets=SETS, score=score, alpha=0.1)
        with pytest.warns(AntecedentDeprecationWarning, match="matched_sets_sensitivity"):
            direct = v.matched_sets_sensitivity(SETS, score=score, alpha=0.1)
        _same(compiled, direct)


def test_the_matched_design_entry_takes_exactly_one_design():
    with pytest.raises(ant.errors.CausalTypeError):
        v.matched_design_sensitivity()
    with pytest.raises(ant.errors.CausalTypeError):
        v.matched_design_sensitivity(PAIRS, sets=SETS)
    with pytest.raises(ant.errors.CausalTypeError):
        v.matched_design_sensitivity(sets=SETS, labels=["a"])


def test_the_supplied_estimate_entries_warn_and_the_compiled_program_reproduces_them():
    for kwargs in (
        {"lower": 1.8, "upper": 8.7, "scale": "risk_ratio"},
        {"lower": 2.25, "upper": 9.0, "scale": "odds_ratio_common"},
        {"scale": "hazard_ratio_common"},
    ):
        compiled = v.supplied_evalue(4.0, **kwargs)
        with pytest.warns(AntecedentDeprecationWarning, match="supplied_evalue"):
            direct = v.evalue(4.0, **kwargs)
        _same(compiled, direct)
    rr = 3.9
    compiled = v.supplied_evalue(rr, lower=1.8, upper=8.7)
    assert compiled.summary["evalue"] == pytest.approx(rr + math.sqrt(rr * (rr - 1.0)), rel=1e-12)

    for model in (None, "mediation", "marginal_structural"):
        compiled = v.supplied_evalue_vector(
            [3.9, 2.0], ["a", "b"], lowers=[1.8, 1.2], uppers=[8.7, 3.3], model=model
        )
        with pytest.warns(AntecedentDeprecationWarning, match="supplied_evalue_vector"):
            direct = v.evalue_vector(
                [3.9, 2.0], ["a", "b"], lowers=[1.8, 1.2], uppers=[8.7, 3.3], model=model
            )
        _same(compiled, direct)

    draws = [[math.exp(0.3 + 0.002 * i) for i in range(400)]]
    for model in (None, "marginal_structural"):
        compiled = v.supplied_evalue_posterior(draws, ["a"], scale="odds_ratio_rare", model=model)
        with pytest.warns(AntecedentDeprecationWarning, match="supplied_evalue_posterior"):
            direct = v.evalue_posterior(draws, ["a"], scale="odds_ratio_rare", model=model)
        _same(compiled, direct)


def test_a_supplied_estimate_refusal_is_the_direct_refusal():
    with (
        pytest.warns(AntecedentDeprecationWarning),
        pytest.raises(ant.CausalError, match="positive") as direct,
    ):
        v.evalue(-1.0, lower=-2.0, upper=0.5)
    with pytest.raises(ant.CausalError, match="positive") as compiled:
        v.supplied_evalue(-1.0, lower=-2.0, upper=0.5)
    assert type(compiled.value) is type(direct.value)
    assert str(compiled.value) == str(direct.value)


def _regime(n=400, seed=45):
    rng = np.random.default_rng(seed)
    l1 = rng.normal(size=n)
    p1 = np.where(l1 > 0.0, 0.7, 0.3)
    a1 = rng.random(n) < p1
    l2 = 0.5 * a1 + rng.normal(size=n)
    p2 = np.where(l2 > 0.0, 0.65, 0.35)
    a2 = rng.random(n) < p2
    y = 1.0 + 2.0 * a1 + 1.5 * a2 + 0.8 * l1 + rng.normal(size=n)
    query = regimes.LongitudinalRegime(
        outcome="y",
        treatment_history=np.column_stack([a1, a2]).tolist(),
        actions=[False, False],
        treatment_probabilities=np.column_stack([p1, p2]).tolist(),
        censoring_probabilities=np.ones((n, 2)).tolist(),
        subject_ids=[f"s{i}" for i in range(n)],
        fold_ids=[i % 5 for i in range(n)],
        method="marginal_structural_model",
        stabilizing_numerator_probabilities=[0.5, 0.5],
        covariates=["l1", "l2"],
    )
    data = {"y": y, "l1": l1, "l2": l2}
    return ant.prepare(data, query=query, refute="none", execution=ant.Execution(seed=7))


def _same_numbers(compiled, direct):
    assert compiled.range is not None and direct.range is not None
    assert compiled.range.assumption == direct.range.assumption
    assert compiled.range.tested == direct.range.tested
    assert compiled.range.break_even == direct.range.break_even
    assert compiled.range.break_even_significance == direct.range.break_even_significance


def test_the_msm_posterior_entries_warn_and_the_compiled_regime_posterior_reproduces_them():
    study = _regime()
    labels = ["period[0]", "period[1]"]
    compiled = v.sensitivity_range(study, "msm_parameter_evalue_posterior", level=0.9, draws=500)
    coefficients = compiled.summary["coefficients"]
    sigma = compiled.summary["sigma"]
    standardized = [[b / s for b, s in zip(column, sigma, strict=True)] for column in coefficients]
    with pytest.warns(AntecedentDeprecationWarning, match="msm_parameter_evalue_posterior"):
        direct = v.evalue_posterior(
            standardized,
            labels,
            scale="standardized_mean_difference",
            level=0.9,
            model="marginal_structural",
        )
    for key in ("median_risk_ratio", "credible_interval", "evalue", "evalue_limit", "retention"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)

    compiled = v.sensitivity_range(
        study, "msm_parameters_ovb_posterior", level=0.9, r2yz_extreme=0.5, draws=500
    )
    sigma = compiled.summary["sigma"]
    bias_scale = [[s / scale for s in sigma] for scale in compiled.summary["exposure_scale"]]
    with pytest.warns(AntecedentDeprecationWarning, match="msm_parameters_ovb_posterior"):
        direct = v.omitted_variable_bias_posterior(
            compiled.summary["coefficients"],
            bias_scale,
            labels,
            level=0.9,
            r2yz_extreme=0.5,
            model="marginal_structural",
        )
    _same_numbers(compiled, direct)
    # The posterior is centred at the route's weighted least-squares fit.
    fit = v.sensitivity_range(study, "msm_parameters")
    for t in range(2):
        mean = sum(compiled.summary["coefficients"][t]) / 500
        assert mean == pytest.approx(fit.summary["estimate"][t], abs=0.25)
