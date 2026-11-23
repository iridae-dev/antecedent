"""Exact transport grids refuse sampling controls they would otherwise ignore."""

import pytest
from antecedent import _native

from test_transport_exact import fixture


def test_statistical_false_refuses_estimator():
    identified, catalog, data = fixture()
    with pytest.raises(Exception, match="statistical grid"):
        _native.prepare_transport_grid(
            identified._native,
            catalog,
            data,
            [{"x": 1.0}],
            statistical=False,
            estimator="bayesian",
        )
