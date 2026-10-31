"""Graph-posterior preparation refuses a target population the query cannot carry."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _native

_TREATED = {"target_population": {"kind": "treated"}}


@pytest.mark.parametrize(
    ("kind", "arguments"),
    [
        (
            "change_attribution",
            {"outcome": "y", "baseline_end": 4, "comparison_start": 4, "comparison_end": 8},
        ),
        ("anomaly_attribution", {"targets": ["y"]}),
        ("nested_counterfactual", {"treatment": "t", "outcome": "y", "mediators": ["m"]}),
    ],
)
def test_non_default_population_refused(kind, arguments):
    columns = [np.arange(8.0) for _ in range(3)]
    with pytest.raises(Exception, match="all-observed population"):
        _native.PreparedAnalysis.prepare_graph_posterior_kind(
            ["t", "m", "y"], columns, kind, options=_TREATED, **arguments
        )
