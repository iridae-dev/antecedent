#!/usr/bin/env python3
"""What are the main effects and the interaction in a 2x2 factorial experiment?

Data: a hand-built balanced fixture of eight rows, two per cell, with cell means 1, 4, 3 and 10,
declared as randomized by an independent fair coin on each factor. The script asserts the main
effects (4 and 5) and the interaction (4) to 1e-9.

`antecedent.factorial.estimate` is a point utility: it reports cell means, effects and
covariance-free variance bounds, and no confidence interval. Pointwise intervals for a fixed-cell
2x2 design come from ``antecedent.experiment.FactorialRandomization`` in a ``RandomizedEffect``
query, the ``factorial_2x2`` row of docs/graphless-support-matrix.md.

Run: `python examples/python/factorial_cell_means.py`."""

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
    # Main effects average over the other factor: A = ((3 - 1) + (10 - 4)) / 2 = 4,
    # B = ((4 - 1) + (10 - 3)) / 2 = 5; the interaction is the difference (10 - 4) - (3 - 1) = 4.
    assert abs(result.factor_a_effect - 4.0) < 1e-9
    assert abs(result.factor_b_effect - 5.0) < 1e-9
    assert abs(result.interaction_effect - 4.0) < 1e-9


if __name__ == "__main__":
    main()
