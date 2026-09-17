#!/usr/bin/env python3
"""Use the first batch of results as prior evidence for a second batch.

We estimate an average effect on batch A, save its posterior, and pass it to
Bayesian(prior_from=...) for an independent batch B. Both batches use the
same graph and model design, so their coefficient positions match.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent import AverageEffect, Bayesian, analyze


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
        refute="none",
        seed=11,
    )
    assert batch_a.posterior is not None
    # Prior hydration needs the posterior payload; batch_a.export() saves the full execution.
    artifact = batch_a.study.export_artifact()

    batch_b = analyze(
        data_b,
        graph=edges,
        query=query,
        inference=Bayesian(n_draws=128, prior_from=artifact),
        refute="none",
        seed=12,
    )
    print("Calibration:", batch_b.calibration.status)
    assert batch_b.study.estimate().posterior is not None
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
