"""A response source whose methods fail or mistype is an error, not a default."""

from __future__ import annotations

import json

import pytest
from antecedent import _native

from test_adaptive_policy import TreatmentSource
from test_adaptive_policy_json import _config


class _Broken(TreatmentSource):
    def __init__(self, method, replacement):
        self._method = method
        self._replacement = replacement
        super().__init__()

    def __getattribute__(self, name):
        if name == object.__getattribute__(self, "_method"):
            return object.__getattribute__(self, "_replacement")
        return super().__getattribute__(name)


def _raises():
    raise RuntimeError("source failure")


@pytest.mark.parametrize(
    ("method", "replacement"),
    [
        ("horizon", _raises),
        ("identity", _raises),
        ("n_draws", lambda split: _raises()),
        ("observable_variables", lambda: ["x"]),
        ("standing", lambda: "fitted"),
    ],
)
def test_source_method_failure_is_an_error(method, replacement):
    with pytest.raises(ValueError, match=method):
        _native.adaptive_policy_search(json.dumps(_config()), _Broken(method, replacement))
