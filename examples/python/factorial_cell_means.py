#!/usr/bin/env python3
"""Point utility for a 2x2 Bernoulli factorial experiment.

`antecedent.factorial.estimate` reports cell means, main effects, the
interaction, and covariance-free variance bounds for a 2x2 factorial randomized
under independent Bernoulli assignment of each factor. It is a point utility: it
carries no confidence interval and has no graphless support-matrix row.

The fixture below is balanced with known cell means, so the main effects (4 and
5) and the interaction (4) are exact. Install with
`python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent import factorial


def main() -> None:
    factor_a = [False, False, False, False, True, True, True, True]
    factor_b = [False, False, True, True, False, False, True, True]
    outcome = np.array([1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 10.0, 10.0])

    result = factorial.estimate(
        {"y": outcome},
        factor_a=factor_a,
        factor_b=factor_b,
        design=factorial.FactorialDesign(0.5, 0.5),
        outcome="y",
    )

    print("cell means:", result.cell_means)
    print("cell support:", result.cell_support)
    print(
        f"A effect={result.factor_a_effect} B effect={result.factor_b_effect} "
        f"interaction={result.interaction_effect}"
    )
    print("uncertainty:", result.uncertainty_semantics)
    # The balanced fixture has exact main effects (4, 5) and interaction (4).
    assert abs(result.factor_a_effect - 4.0) < 1e-9
    assert abs(result.factor_b_effect - 5.0) < 1e-9
    assert abs(result.interaction_effect - 4.0) < 1e-9


if __name__ == "__main__":
    main()
