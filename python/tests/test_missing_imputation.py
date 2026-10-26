"""Multiple imputation through ``missing=``: declaration, pooling, recovery, refusals."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalTypeError, CausalValueError
from antecedent.missing import ImputationSpec, MissingPolicy, coerce_missing, native_declaration
from antecedent.outcomes import require_success

TRUE_EFFECT = 1.0
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _data(n: int, seed: int):
    """``y = t + 0.7 z + e``; the confounder ``z`` is missing with probability rising in
    ``y`` (missing at random given the outcome). Complete cases select on the outcome and
    attenuate the effect to about 0.73; imputation with ``y`` in the model recovers it."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = TRUE_EFFECT * t + 0.7 * z + rng.normal(size=n)
    missing = rng.random(n) < 1 / (1 + np.exp(-(1.5 * y - 0.5)))
    return {"t": t, "y": y, "z": np.where(missing, np.nan, z)}


def _analyze(data, missing):
    return require_success(
        ant.analyze(
            data,
            query=ant.AverageEffect("t", "y"),
            graph=GRAPH,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
            missing=missing,
        )
    )


def test_the_declaration_is_validated_and_carried_to_the_native_option():
    policy = MissingPolicy.multiple_imputation(imputations=12, seed=4, auxiliary=("w",))
    assert policy.policy == "multiple_imputation"
    assert policy.imputation == ImputationSpec(imputations=12, seed=4)
    assert policy.covariates == ("w",)
    assert coerce_missing("multiple_imputation").imputation == ImputationSpec()
    native = native_declaration(policy, ["t", "y", "w"])
    assert native == {
        "policy": "multiple_imputation",
        "covariates": [2],
        "imputation": {
            "model": "chained_equations",
            "imputations": 12,
            "iterations": 10,
            "seed": 4,
            "assumption": "missing_at_random",
        },
    }
    with pytest.raises(CausalValueError):
        ImputationSpec(model="hot_deck")
    with pytest.raises(CausalValueError):
        ImputationSpec(assumption="unknown")
    with pytest.raises(CausalTypeError):
        ImputationSpec(imputations=-1)
    with pytest.raises(CausalValueError):
        MissingPolicy("complete_case", imputation=ImputationSpec())


def test_imputation_recovers_the_effect_that_complete_cases_attenuate():
    imputed, complete = [], []
    for seed in range(4):
        data = _data(1_500, seed)
        policy = MissingPolicy.multiple_imputation(imputations=10, iterations=5, seed=seed)
        result = _analyze(data, policy)
        imputed.append(result.estimate.ate)
        complete.append(_analyze(data, "complete_case").estimate.ate)
        report = result.missing_data
        assert report.policy == "multiple_imputation"
        assert report.accounting.rows_dropped == 0
        pooled = report.imputation
        assert pooled is not None and pooled.pooling == "rubin"
        assert pooled.imputations == 10 and len(pooled.points) == 10
        assert pooled.imputed_cells["z"] > 0
        assert pooled.total == pytest.approx(pooled.within + 1.1 * pooled.between)
        assert result.estimate.ate == pytest.approx(np.mean(pooled.points))
    assert np.mean(imputed) == pytest.approx(TRUE_EFFECT, abs=0.08)
    assert abs(np.mean(complete) - TRUE_EFFECT) > 0.15


def test_the_result_is_deterministic_in_the_seed():
    data = _data(600, 11)
    first = _analyze(data, MissingPolicy.multiple_imputation(imputations=4, seed=2))
    again = _analyze(data, MissingPolicy.multiple_imputation(imputations=4, seed=2))
    other = _analyze(data, MissingPolicy.multiple_imputation(imputations=4, seed=3))
    assert first.estimate.ate == again.estimate.ate
    assert first.missing_data.imputation.points == again.missing_data.imputation.points
    assert first.missing_data.imputation.points != other.missing_data.imputation.points


def test_unsupported_declarations_are_refused_with_the_next_attempt():
    data = _data(300, 5)
    with pytest.raises(CausalError) as not_at_random:
        _analyze(data, MissingPolicy.multiple_imputation(assumption="missing_not_at_random"))
    assert getattr(not_at_random.value, "reason_code", None) == "missing_policy_not_applicable"
    assert "sensitivity analysis" in str(not_at_random.value)
    with pytest.raises(CausalError) as panel:
        _analyze(data, "balanced_panel")
    assert getattr(panel.value, "reason_code", None) == "missing_policy_not_applicable"
    assert "next attempt: declare `multiple_imputation`" in str(panel.value)
