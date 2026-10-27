#!/usr/bin/env python3
"""Use the first batch of results as prior evidence for a second batch.

We estimate an average effect on batch A, save its posterior, and pass it to
Bayesian(prior_from=...) for an independent batch B. Both batches use the
same graph and model design, so their coefficient positions match.

Both batches are simulated from one declared mechanism, y = 2 t + z + noise, so
the true average effect of t is 2.0. That is a fixture truth of the generator.
The check is that each posterior mean sits near 2.0 and that the second
posterior is tighter than the first, because it starts from batch A's evidence.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import antecedent as _ant_api
import numpy as np
from antecedent import AverageEffect, Bayesian, analyze
from antecedent.outcomes import require_success


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

    outcome_a = analyze(
        data_a,
        graph=edges,
        query=query,
        inference=Bayesian(n_draws=128),
        refute="none",
        execution=_ant_api.Execution(seed=11),
    )
    batch_a = require_success(outcome_a)
    assert batch_a.posterior is not None
    # Prior hydration needs the posterior payload; a plain export would save the full execution.
    artifact = bytes(outcome_a.export(quantity="posterior"))

    outcome_b = analyze(
        data_b,
        graph=edges,
        query=query,
        inference=Bayesian(n_draws=128, prior_from=artifact),
        refute="none",
        execution=_ant_api.Execution(seed=12),
    )
    batch_b = require_success(outcome_b)
    print("Calibration:", batch_b.calibration.status)
    assert require_success(_ant_api.estimate(outcome_b.study)).posterior is not None
    assert batch_b.posterior is not None
    assert np.isfinite(batch_b.posterior.effect_mean)
    assert batch_b.identification.assumption_count >= 1
    # Declared truth: the effect of t is 2.0 in both batches.
    assert abs(batch_a.posterior.effect_mean - 2.0) < 0.2, batch_a.posterior.effect_mean
    assert abs(batch_b.posterior.effect_mean - 2.0) < 0.2, batch_b.posterior.effect_mean
    # Carrying batch A's posterior forward as the prior narrows batch B's.
    assert batch_b.posterior.effect_sd < batch_a.posterior.effect_sd, (
        batch_b.posterior.effect_sd,
        batch_a.posterior.effect_sd,
    )
    print(
        f"A effect_mean={batch_a.posterior.effect_mean:.4f} "
        f"B effect_mean={batch_b.posterior.effect_mean:.4f} "
        f"assumptions={batch_b.identification.assumption_count}"
    )


if __name__ == "__main__":
    main()
