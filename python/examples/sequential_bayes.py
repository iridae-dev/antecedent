#!/usr/bin/env python3
"""Sequential Bayes: batch A posterior → batch B prior.

Requires a built causal extension (`maturin develop` in python/).

Fits Bayesian ATE on batch A, encodes the posterior artifact, then re-analyzes
an independent batch B with ``Bayesian(prior_from=artifact)`` on the same
graph/design (index-aligned coefficient hydrate).
"""

from __future__ import annotations

import numpy as np

from causal import AverageEffect, Bayesian, analyze


def _batch(n: int, seed: int) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.4 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def main() -> None:
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    data_a, _ = _batch(180, seed=1)
    data_b, _ = _batch(180, seed=2)
    query = AverageEffect(treatment="t", outcome="y")

    batch_a = analyze(
        data_a,
        graph=edges,
        query=query,
        inference=Bayesian(n_draws=128),
        refute=False,
        seed=11,
        return_posterior_artifact=True,
    )
    assert batch_a.posterior is not None
    artifact = bytes(batch_a.posterior.artifact)

    batch_b = analyze(
        data_b,
        graph=edges,
        query=query,
        inference=Bayesian(n_draws=128, prior_from=artifact),
        refute=False,
        seed=12,
    )
    assert batch_b.posterior is not None
    assert np.isfinite(batch_b.posterior.effect_mean)
    assert batch_b.identification.assumption_count >= 1
    print(
        f"A effect_mean={batch_a.posterior.effect_mean:.4f} "
        f"B effect_mean={batch_b.posterior.effect_mean:.4f} "
        f"assumptions={batch_b.identification.assumption_count}"
    )


if __name__ == "__main__":
    main()
