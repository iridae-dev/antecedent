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
    with pytest.raises(CausalValueError, match="unknown imputation model 'hot_deck'"):
        ImputationSpec(model="hot_deck")
    with pytest.raises(CausalValueError, match="unknown missingness assumption 'unknown'"):
        ImputationSpec(assumption="unknown")
    with pytest.raises(CausalTypeError, match="imputations must be a non-negative integer"):
        ImputationSpec(imputations=-1)
    with pytest.raises(CausalValueError, match="read only by 'multiple_imputation'"):
        MissingPolicy("complete_case", imputation=ImputationSpec())


@pytest.mark.parametrize("name,value", [("imputations", 0), ("imputations", 1),
                                         ("imputations", 1001), ("iterations", 0),
                                         ("iterations", 1001)])
def test_imputation_counts_respect_documented_bounds(name, value):
    with pytest.raises(CausalValueError, match=name):
        ImputationSpec(**{name: value})
    from antecedent.missing import multiple_imputation_with_graph_selection

    with pytest.raises(CausalValueError, match=name):
        multiple_imputation_with_graph_selection(
            {"t": [0, 1], "y": [1, 2]}, treatment="t", outcome="y", **{name: value}
        )


def test_imputation_counts_accept_documented_endpoints():
    assert ImputationSpec(imputations=2, iterations=1).imputations == 2
    assert ImputationSpec(imputations=1000, iterations=1000).iterations == 1000


def test_graph_selection_converts_pandas_na_before_native_call(monkeypatch):
    pd = pytest.importorskip("pandas")
    from antecedent import _native
    from antecedent.missing import multiple_imputation_with_graph_selection

    class Captured(Exception):
        pass

    def capture(names, columns, *args, **kwargs):
        assert names == ["t", "y"]
        assert np.isnan(columns[1][1])
        assert kwargs["domains"] == [("y", "binary")]
        raise Captured

    monkeypatch.setattr(_native, "graph_selection_imputation_run", capture)
    with pytest.raises(Captured):
        multiple_imputation_with_graph_selection(
            {"t": [0, 1], "y": [1, pd.NA]}, treatment="t", outcome="y",
            domains=(("y", "binary"),),
        )


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


def _analyze_with(data, missing, estimator):
    return require_success(
        ant.analyze(
            data,
            query=ant.AverageEffect("t", "y"),
            graph=GRAPH,
            refute="none",
            estimator=estimator,
            inference=ant.Frequentist(bootstrap=0),
            missing=missing,
        )
    )


@pytest.mark.parametrize(
    "estimator", ["propensity.matching", "propensity.stratification", "distance.matching"]
)
def test_matching_and_stratification_estimators_are_pooled_by_rubins_rules(estimator):
    data = _data(900, 3)
    policy = MissingPolicy.multiple_imputation(imputations=6, iterations=4, seed=2)
    result = _analyze_with(data, policy, estimator)
    pooled = result.missing_data.imputation
    assert pooled is not None and pooled.pooling == "rubin"
    assert pooled.imputations == 6 and len(pooled.points) == 6
    # Each completed table refits the whole estimator, so the tables' estimates differ.
    assert pooled.between > 0.0
    assert result.estimate.ate == pytest.approx(np.mean(pooled.points))
    assert pooled.within > 0.0
    assert pooled.total == pytest.approx(pooled.within + (1.0 + 1.0 / 6.0) * pooled.between)
    lam = (1.0 + 1.0 / 6.0) * pooled.between / pooled.total
    assert pooled.df == pytest.approx(5.0 / lam**2, rel=1e-6)
    # Observed errors are 0.03 to 0.04; the complete-case estimate of the same table is
    # attenuated to about 0.72-0.78, which a 0.15 band excludes.
    assert result.estimate.ate == pytest.approx(TRUE_EFFECT, abs=0.15)


def test_an_estimator_without_a_pooled_variance_is_refused_with_the_next_attempt():
    data = _data(300, 5)
    with pytest.raises(CausalError) as refused:
        _analyze_with(data, MissingPolicy.multiple_imputation(seed=1), "dml")
    assert getattr(refused.value, "reason_code", None) == "missing_policy_not_applicable"
    assert "next attempt: declare `complete_case`" in str(refused.value)


def test_the_within_table_bootstrap_setting_is_declared_and_carried_to_the_native_option():
    policy = MissingPolicy.multiple_imputation(imputations=4, bootstrap_replicates=60)
    assert policy.imputation == ImputationSpec(imputations=4, bootstrap_replicates=60)
    native = native_declaration(policy, ["t", "y"])
    assert native["imputation"]["bootstrap_replicates"] == 60
    # Absent when unset, so the native option of an ordinary declaration is unchanged.
    plain = native_declaration(MissingPolicy.multiple_imputation(), ["t", "y"])
    assert "bootstrap_replicates" not in plain["imputation"]
    with pytest.raises(CausalTypeError, match="bootstrap_replicates must be a non-negative"):
        ImputationSpec(bootstrap_replicates=-1)


def _selection_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = TRUE_EFFECT * t + 0.7 * z + rng.normal(size=n)
    miss = rng.random(n) < 1 / (1 + np.exp(-(1.2 * z - 0.8)))
    return {"t": t, "y": np.where(miss, np.nan, y), "z": z}


def test_graph_selection_is_rerun_on_every_completed_table():
    from antecedent.missing import (
        GraphSelectionImputationRefusal,
        multiple_imputation_with_graph_selection,
    )

    data = _selection_data(500, 3)
    data["t"][::10] = np.nan
    result = multiple_imputation_with_graph_selection(
        data, treatment="t", outcome="y", imputations=4, domains=(("t", "binary"),),
        iterations=3,
        n_draws=200,
        execution=ant.Execution(seed=2),
    )
    assert result.structure == "graph_selection_rerun_per_imputation"
    assert result.pooling == "stacked_graph_posterior_rerun"
    assert result.estimator == "bayesian.gcomp"
    assert result.assumption == "missing_at_random"
    assert result.imputations == 4 and len(result.points) == 4
    assert result.posterior_draws == 4 * 200
    assert result.imputed_cells["y"] > 0
    assert result.imputed_cells["t"] == 50
    assert result.domains == {"t": "binary"}
    assert result.binary_boundaries == {}
    # The stack's mean is the mean of the tables' posterior means, up to Monte Carlo error.
    assert result.effect == pytest.approx(np.mean(result.points), abs=8 * result.posterior_sd / 28)
    lower, upper = result.credible_interval["lower"], result.credible_interval["upper"]
    assert lower < result.effect < upper
    assert result.total >= result.within and result.between >= 0.0

    with pytest.raises(GraphSelectionImputationRefusal) as unknown:
        multiple_imputation_with_graph_selection(data, treatment="t", outcome="nope")
    assert unknown.value.detail == "graph_selection_imputation.variables"


def test_explicit_constant_domains_are_carried_and_change_identity():
    data = _data(300, 71)
    data["w"] = np.zeros(300)
    data["w"][::10] = np.nan
    # With an explicit auxiliary list, include the missing confounder as well as
    # the constant column: only queried variables and named auxiliaries are imputed.
    kwargs = dict(imputations=3, iterations=2, seed=3, auxiliary=("z", "w"))
    binary = MissingPolicy.multiple_imputation(**kwargs, domains=(("w", "binary"),))
    continuous = MissingPolicy.multiple_imputation(**kwargs, domains=(("w", "continuous"),))
    assert native_declaration(binary, data.keys())["imputation"]["domains"] == [("w", "binary")]
    assert binary.imputation.domains == (("w", "binary"),)
    with pytest.raises(CausalError, match="ambiguous_variable_domain|declare its domain"):
        _analyze(data, MissingPolicy.multiple_imputation(**kwargs))
    a = _analyze(data, binary)
    b = _analyze(data, continuous)
    assert a.missing_data.imputation.assumptions_digest != b.missing_data.imputation.assumptions_digest
    assert a.missing_data.imputation.domains == {"w": "binary"}
    assert a.missing_data.imputation.binary_boundaries == {"w": {"class": 0, "method": "binary_single_class_empirical_boundary_v1"}}
    assert b.missing_data.imputation.domains == {"w": "continuous"}
    assert not b.missing_data.imputation.binary_boundaries
    with pytest.raises(CausalError, match="binary_under_joint_normal|binary"):
        _analyze(data, MissingPolicy.multiple_imputation(**kwargs, model="joint_normal", domains=(("w", "binary"),)))
    with pytest.raises(CausalValueError):
        ImputationSpec(domains=(("w", "binary"), ("w", "continuous")))
