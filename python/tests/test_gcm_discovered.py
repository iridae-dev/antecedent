"""fit_gcm_discovered compose helper."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def test_fit_gcm_discovered_lingam_smoke():
    n = 400
    rng = np.random.default_rng(4)
    # Non-Gaussian noise so LiNGAM can orient.
    e_z = rng.uniform(-1.0, 1.0, size=n)
    e_t = rng.uniform(-1.0, 1.0, size=n)
    e_y = rng.uniform(-1.0, 1.0, size=n)
    z = e_z
    t = 0.8 * z + e_t
    y = 1.5 * t + 0.6 * z + e_y
    data = {"z": z, "t": t, "y": y}
    fitted, edges = causal.fit_gcm_discovered(
        data, discovery=causal.LiNGAM(), seed=1
    )
    assert edges
    assert fitted is not None


def test_fit_gcm_discovered_refuses_fci_and_incomplete_pc():
    n = 80
    rng = np.random.default_rng(5)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = t + z + rng.normal(size=n) * 0.3
    data = {"z": z, "t": t, "y": y}
    with pytest.raises(ValueError, match="fully oriented"):
        causal.fit_gcm_discovered(data, discovery=causal.FCI(alpha=0.2, fdr=False))
    # Weak PC often leaves undirected marks — compose must fail closed.
    with pytest.raises(ValueError, match="cannot coerce|incomplete|orient"):
        causal.fit_gcm_discovered(
            data, discovery=causal.PC(alpha=0.5, fdr=False, max_cond_size=0), seed=1
        )
