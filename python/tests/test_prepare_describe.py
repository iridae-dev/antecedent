"""prepare() validates the refusal handling it is given and never drops it."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.estimation import _PreparedEngine


def _data(n: int = 60):
    rng = np.random.default_rng(3)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(float)
    return {"z": z, "t": t, "y": 2.0 * t + z}


def test_on_refusal_typo_refused():
    with pytest.raises(CausalValueError, match="on_refusal"):
        _PreparedEngine.prepare(
            _data(),
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ant.AverageEffect("t", "y"),
            on_refusal="describ",
        )


def test_candidates_without_describe_refused():
    with pytest.raises(CausalUnsupportedError, match="evidence_candidates"):
        _PreparedEngine.prepare(
            _data(),
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ant.AverageEffect("t", "y"),
            evidence_candidates=[object()],
        )


