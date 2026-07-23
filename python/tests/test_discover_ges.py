"""GES screen_pc / max_subset surface."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def test_discover_ges_screen_pc_smoke():
    n = 200
    rng = np.random.default_rng(11)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.4
    y = 1.2 * t + z + rng.normal(size=n) * 0.4
    result = antecedent.discover_ges(
        data={"t": t, "y": y, "z": z},
        alpha=0.2,
        fdr=False,
        screen_pc=True,
        max_subset=4,
        seed=1,
    )
    assert result.cpdag_nodes == 3
    assert isinstance(result.cpdag_directed_edges, int)
