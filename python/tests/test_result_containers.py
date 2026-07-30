"""Container dunders on the views that are conceptually collections.

Covers ``ValidationView`` (len/iter/getitem/.failed/.to_pandas),
``PosteriorView`` (__array__/.interval), and ``IdentificationView.__bool__``
edge cases beyond the happy path already covered in test_result_repr.py.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from antecedent.results import (
    IdentificationView,
    PosteriorView,
    RefutationReport,
    ValidationView,
)


def _refutation(refuter: str, passed: bool) -> RefutationReport:
    return RefutationReport(
        refuter=refuter,
        original_ate=1.0,
        refuted_ate=0.01 if passed else 0.9,
        comparison=0.1,
        informative=True,
        passed=passed,
        failure_condition=None if passed else "too close",
        replicates=50,
    )


# --- IdentificationView.__bool__ -----------------------------------------------------


def test_identification_view_bool_strips_whitespace_and_case():
    assert bool(
        IdentificationView(
            status="  NonparametricallyIdentified  ",
            method="m",
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        )
    )
    assert bool(
        IdentificationView(
            status="GCM.PARAMETRIC",
            method="m",
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        )
    )


# --- ValidationView -----------------------------------------------------


def test_validation_view_len_iter_getitem():
    reports = [_refutation("placebo", True), _refutation("bootstrap", False)]
    view = ValidationView(passed=False, ran=True, count=2, reports=reports)

    assert len(view) == 2
    assert list(view) == reports
    assert view[0] is reports[0]
    assert view[-1] is reports[1]
    assert view["bootstrap"] is reports[1]


def test_validation_view_getitem_missing_key_raises_keyerror():
    view = ValidationView(passed=True, ran=True, count=1, reports=[_refutation("placebo", True)])
    with pytest.raises(KeyError):
        view["does_not_exist"]


def test_validation_view_getitem_out_of_range_raises_indexerror():
    view = ValidationView(passed=True, ran=True, count=1, reports=[_refutation("placebo", True)])
    with pytest.raises(IndexError):
        view[5]


def test_validation_view_empty_is_falsy_length():
    view = ValidationView(passed=False, ran=False, count=0)
    assert len(view) == 0
    assert list(view) == []


def test_validation_view_failed_property():
    reports = [_refutation("placebo", True), _refutation("bootstrap", False)]
    view = ValidationView(passed=False, ran=True, count=2, reports=reports)
    assert view.failed == [reports[1]]


def test_validation_view_failed_property_all_passed():
    reports = [_refutation("placebo", True), _refutation("bootstrap", True)]
    view = ValidationView(passed=True, ran=True, count=2, reports=reports)
    assert view.failed == []


def test_validation_view_to_pandas_happy_path():
    pd = pytest.importorskip("pandas")
    reports = [_refutation("placebo", True), _refutation("bootstrap", False)]
    view = ValidationView(passed=False, ran=True, count=2, reports=reports)
    frame = view.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert list(frame["refuter"]) == ["placebo", "bootstrap"]
    assert list(frame["passed"]) == [True, False]


def test_validation_view_to_pandas_missing_dependency_raises_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    view = ValidationView(passed=True, ran=True, count=0, reports=[])
    with pytest.raises(ImportError, match="pandas"):
        view.to_pandas()


# --- PosteriorView.__array__ -----------------------------------------------------


def test_posterior_view_array_raises_without_artifact():
    view = PosteriorView(
        effect_mean=1.0,
        effect_sd=0.1,
        q025=0.8,
        q975=1.2,
        n_draws=5,
        p_below_zero=0.1,
        backend="conjugate",
        artifact=None,
    )
    with pytest.raises(ValueError, match="artifact"):
        np.asarray(view)


def test_posterior_view_array_decodes_real_artifact():
    native = pytest.importorskip("antecedent._native")
    artifact = native.PosteriorArtifact.from_moments(
        n_draws=5,
        mean=[1.0],
        sd=[0.1],
        q025=[0.8],
        q975=[1.2],
        backend_id="conjugate",
        identification="identified",
        quantity_names=["ate"],
    )
    encoded = native.encode_posterior_artifact(artifact)
    view = PosteriorView(
        effect_mean=1.0,
        effect_sd=0.1,
        q025=0.8,
        q975=1.2,
        n_draws=5,
        p_below_zero=0.1,
        backend="conjugate",
        artifact=encoded,
    )
    arr = np.asarray(view)
    assert isinstance(arr, np.ndarray)


# --- PosteriorView.interval -----------------------------------------------------


def test_posterior_view_interval_default_and_explicit_95():
    view = PosteriorView(
        effect_mean=1.0,
        effect_sd=0.1,
        q025=0.8,
        q975=1.2,
        n_draws=5,
        p_below_zero=0.1,
        backend="x",
    )
    assert view.interval() == (0.8, 1.2)
    assert view.interval(level=0.95) == (0.8, 1.2)


def test_posterior_view_interval_other_level_raises_value_error():
    view = PosteriorView(
        effect_mean=1.0,
        effect_sd=0.1,
        q025=0.8,
        q975=1.2,
        n_draws=5,
        p_below_zero=0.1,
        backend="x",
    )
    with pytest.raises(ValueError, match="0.95"):
        view.interval(level=0.90)


def test_posterior_view_interval_missing_quantiles_raises():
    view = PosteriorView(
        effect_mean=None,
        effect_sd=None,
        q025=None,
        q975=None,
        n_draws=None,
        p_below_zero=None,
        backend=None,
    )
    with pytest.raises(ValueError):
        view.interval()
