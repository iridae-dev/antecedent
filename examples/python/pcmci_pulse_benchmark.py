#!/usr/bin/env python3
"""Discover a known lag-1 SCM with PCMCI, then estimate a PulseEffect.

Mirrors the Exact parent-set conformance fixture at
``conformance/discovery/pcmci_lag1`` (see ``expected.json`` / baseline pin
``parity/baselines/tigramite.toml``):

  x_t ~ N(0, 1)
  y_t = 0.8 * x_{t-1} + 0.2 * N(0, 1)
  n=500, seed=0, max_lag=2, alpha=0.05, fdr=false

Expected edge recovery (Exact): lagged parent ``(x, 1) → (y, 0)``.
Expected pulse (treatment ``x`` → outcome ``y``, treatment_lag=1,
horizon_steps=1): ≈ 0.8 within absolute tolerance 0.15.

Black-box only: Antecedent's PCMCI path; do not vendor or translate Tigramite.
Install with `python -m pip install antecedent`; see examples/README.md.
"""

from __future__ import annotations

import numpy as np

import antecedent

TRUE_PARENTS = {("x", 1, "y", 0)}
TRUE_PULSE = 0.8
# Documented bound for this benchmark narrative (not a statistical CI).
PULSE_TOL = 0.15


def _pcmci_lag1_scm(n: int = 500, seed: int = 0) -> dict[str, np.ndarray]:
    """Conformance-aligned lag-1 linear SCM (pcmci_lag1 expected.json)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).astype(np.float64)
    y = np.empty(n, dtype=np.float64)
    y[0] = 0.2 * rng.normal()
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def main() -> None:
    data = _pcmci_lag1_scm()

    accepted = antecedent.discovery.PCMCI(max_lag=2, alpha=0.05, fdr=False).accept(
        data, seed=0
    )
    assert isinstance(accepted, antecedent.AcceptedGraph)

    recovered = set(accepted.graph.edges())
    print(f"Recovered edges: {sorted(recovered)}")
    print(f"True parents:    {sorted(TRUE_PARENTS)}")
    assert recovered == TRUE_PARENTS, recovered

    result = antecedent.analyze(
        data,
        graph=accepted,
        query=antecedent.PulseEffect(
            treatment="x",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        bootstrap=0,
        seed=0,
        refute=False,
    )
    pulse = float(result.ate)
    print(
        f"PulseEffect={pulse:.4f} truth={TRUE_PULSE} tol={PULSE_TOL} "
        f"plan={result.performance.plan_id}"
    )
    assert abs(pulse - TRUE_PULSE) < PULSE_TOL, pulse


if __name__ == "__main__":
    main()
