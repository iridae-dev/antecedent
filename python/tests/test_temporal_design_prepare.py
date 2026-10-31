"""The temporal design declaration refuses keys it does not read."""

from __future__ import annotations

import json

import pytest
from antecedent import _native

_DOSE = {
    "design": "dose",
    "dose": [0.1, 0.2],
    "density": [1.0, 1.0],
    "bandwidth": 0.5,
    "target": {"kind": "local_quadratic", "grid": [0.1], "derivative": False},
}
_MULTI_ARM = {
    "design": "multi_arm",
    "assignment": [0, 1],
    "probabilities": [[0.5, 0.5], [0.5, 0.5]],
    "arms": ["a", "b"],
}


def _prepare(design):
    return _native.PreparedAnalysis.prepare_temporal_design(
        ["y"], [[1.0, 2.0]], "y", ["u", "u"], [0, 1], json.dumps(design)
    )


@pytest.mark.parametrize(
    ("design", "key"),
    [
        ({**_MULTI_ARM, "ordred": True}, "ordred"),
        ({**_DOSE, "target": {**_DOSE["target"], "curvature_bounds": 0.3}}, "curvature_bounds"),
        ({**_DOSE, "min_probability": 0.2}, "min_probability"),
    ],
)
def test_unknown_design_key_refused(design, key):
    with pytest.raises(ValueError, match=key):
        _prepare(design)
