"""Temporal mediation typed query recovers mediated path product."""

from __future__ import annotations

import numpy as np

import causal


def test_temporal_mediation_decomposition():
    n = 200
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + 0.1 * np.sin(i)
        m[i] = 0.8 * t[i - 1] + 0.05 * np.cos(i)
        # No direct T→Y: mediated-only DGP for MediationContrast::Mediated ID.
        y[i] = 0.5 * m[i] + 0.02 * np.sin(i)
    data = {"t": t, "m": m, "y": y}
    edges = [
        ("t", 1, "t", 0),
        ("t", 1, "m", 0),
        ("m", 0, "y", 0),
    ]
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.mediation is not None
    assert result.mediation.mediated is not None and result.mediation.mediated > 0.1
    if (
        result.mediation.total is not None
        and result.mediation.direct is not None
        and result.mediation.mediated is not None
    ):
        assert abs(
            result.mediation.total - result.mediation.direct - result.mediation.mediated
        ) < 0.15
