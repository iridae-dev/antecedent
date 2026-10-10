"""Public selected Verma identification, independent of the nested likelihood."""

from itertools import product

import antecedent as ac
import numpy as np
import pytest
from antecedent.recalc import Utility
from antecedent.recalc_static import StaticResponseRequest, StaticResponseSession


def test_nested_pilot_selected_public_identification_matches_latent_scm():
    # U is an unobserved common cause of X2 and X4. The observations below
    # enumerate its law; the do-means integrate the intervened SCM directly.
    observed = []
    for u, x1, x2, x3, x4 in product((0, 1), repeat=5):
        probabilities = (
            0.3,
            0.5,
            0.1 + 0.2 * x1 + 0.4 * u,
            0.25 + 0.5 * x2,
            0.15 + 0.35 * x3 + 0.3 * u,
        )
        mass = np.prod(
            [p if bit else 1 - p for p, bit in zip(probabilities, (u, x1, x2, x3, x4), strict=True)]
        )
        repetitions = round(float(mass) * 16_000)
        assert float(mass) * 16_000 == pytest.approx(repetitions, abs=1e-10)
        observed.extend([(x1, x2, x3, x4)] * repetitions)
    rows = np.asarray(observed, dtype=float)
    assert rows.shape == (16_000, 4)
    names = ["X1", "X2", "X3", "X4"]
    graph = ac.Admg.from_edges(names, [("X1", "X2"), ("X2", "X3"), ("X3", "X4")], [("X2", "X4")])
    request = StaticResponseRequest(
        dict(zip(names, rows.T, strict=True)), graph, "X2", "X4", [0, 1], Utility(1)
    )
    result = StaticResponseSession().execute(request, seed=11)
    # Under do(X2=a), E[X3]=.25+.5a, E[U]=.3, so
    # E[X4]=.15+.35(.25+.5a)+.3*.3. No fitted value supplies the oracle.
    assert result.means == pytest.approx((0.3275, 0.5025), abs=1e-12)
    assert result.contrast == pytest.approx(0.175, abs=1e-12)
    assert result.receipt.totals.identifications > 0
    assert result.receipt.totals.factor_builds > 0
