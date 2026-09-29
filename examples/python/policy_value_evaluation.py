#!/usr/bin/env python3
"""Evaluate the held-out value of a fixed binary policy.

`antecedent.policy.evaluate_policy` scores a precomputed binary recommendation
on subjects with known randomized assignment probabilities. It reports the
policy value, the reference (observed-assignment) value, the incremental value,
the treatment rate, and net treatment cost. Under a binding capacity or budget
constraint the scalar value stays point-only.

This evaluates a policy; it does not train one. The recommendations and
propensities are supplied by the caller. Install with
`python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent import policy


def main() -> None:
    assignment = [True, False, True, False]
    outcome = 5.0 + 2.0 * np.asarray(assignment, dtype=np.float64)

    # A fixed binary recommendation with a per-treated cost.
    recommended = policy.BinaryPolicy([True, True, False, False], costs=0.5)

    result = policy.evaluate_policy(
        {"y": outcome},
        outcome="y",
        assignment=assignment,
        propensity=0.5,
        policy=recommended,
    )

    print(f"policy value={result.policy_value}")
    print(f"reference value={result.reference_value}")
    print(f"incremental value={result.incremental_value}")
    print(f"treatment rate={result.treatment_rate} cost={result.total_treatment_cost}")
    print("uncertainty:", result.uncertainty)
    # The incremental value is the policy value net of the observed-assignment reference.
    assert abs(result.incremental_value - (result.policy_value - result.reference_value)) < 1e-9


if __name__ == "__main__":
    main()
