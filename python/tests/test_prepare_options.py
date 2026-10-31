"""Prepare options and explicit arguments do not silently override or drop each other."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _native


def _columns():
    return [np.array([float(i % 2) for i in range(8)]), np.arange(8.0)]


def test_options_identifier_does_not_override_explicit():
    with pytest.raises(ValueError, match="contradicts"):
        _native.PreparedAnalysis.prepare(
            ["t", "y"],
            _columns(),
            [("t", "y")],
            "t",
            "y",
            identifier="backdoor.adjustment",
            options={"identifier": "backdoor.efficient"},
        )


@pytest.mark.parametrize(
    "options",
    [{"sampling_weights": [1.0] * 8}, {"identifier": "backdoor.adjustment"}],
)
def test_observation_route_refuses_unread_declarations(options):
    with pytest.raises(Exception, match="reads no"):
        _native.prepare_observation_response(
            ["t", "y", "latent", "seen"],
            [np.full(8, float(i)) for i in range(4)],
            [("t", "y")],
            "t",
            "y",
            [0.0, 1.0],
            "missing_at_random",
            "latent",
            observed="seen",
            assumption_kind="missing_at_random",
            options=options,
        )
