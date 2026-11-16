#!/usr/bin/env python3
"""Does PCMCI recover a known lagged parent, and what pulse effect follows under that graph?

The series is simulated from a declared lag-1 linear mechanism, the one used by the conformance
fixture ``conformance/discovery/pcmci_lag1``:

  x_t ~ N(0, 1)
  y_t = 0.8 * x_{t-1} + 0.2 * N(0, 1)
  n=500, seed=0, max_lag=2, alpha=0.05, fdr=false

The script asserts that discovery returns exactly the lagged parent ``(x, 1) → (y, 0)`` and that
the pulse effect of ``x`` on ``y`` one step later, estimated under the accepted graph, is within
0.15 of the simulated 0.8.

The run uses Antecedent's own PCMCI implementation; Tigramite is a reference, not a
dependency. Recovering this one parent on one seeded series does not show discovery is reliable
on other data.

Run: ``python examples/python/pcmci_pulse_benchmark.py``.
"""

from __future__ import annotations

import antecedent
import numpy as np

TRUE_PARENTS = {("x", 1, "y", 0)}
TRUE_PULSE = 0.8
# A fixed tolerance for this example, not a statistical interval.
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

    accepted = antecedent.discovery.discover(
        data,
        method=antecedent.discovery.PCMCI(max_lag=2, alpha=0.05, fdr=False),
        accept_discovered=True,
        execution=antecedent.Execution(seed=0),
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
        inference=antecedent.Frequentist(bootstrap=0),
        execution=antecedent.Execution(seed=0),
        refute=False,
    )
    pulse = float(result.answer.value)
    print(
        f"PulseEffect={pulse:.4f} truth={TRUE_PULSE} tol={PULSE_TOL} "
        f"plan={result.value.performance.plan_id}"
    )
    assert abs(pulse - TRUE_PULSE) < PULSE_TOL, pulse


if __name__ == "__main__":
    main()
