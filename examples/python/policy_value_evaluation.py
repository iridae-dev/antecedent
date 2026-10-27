#!/usr/bin/env python3
"""Evaluate the held-out value of a fixed binary policy.

`antecedent.policy.evaluate_policy` scores a precomputed binary recommendation
on subjects with known randomized assignment probabilities. It reports the
policy value, the reference (observed-assignment) value, the incremental value,
the treatment rate, and net treatment cost. Under a binding capacity or budget
constraint the scalar value stays point-only.

The fixture is simulated with a declared mechanism: treating raises the outcome
by 2.0 from a baseline of 5.0, assignment is randomized at probability 0.5, and
the recommendation treats half the rows at a cost of 0.5 each. The known truth
is a fixture truth of that mechanism: treating half the population gains
0.5 * (2.0 - 0.5) = 0.75 over treating no one.

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
    # The incremental value is the policy value net of the reference value.
    assert (
        abs(result.incremental_value - (result.policy_value - result.reference_value))
        < 1e-9
    )
    # Known truth: effect 2.0, cost 0.5 per treated row, half the rows treated.
    treated_share, effect, unit_cost = 0.5, 2.0, 0.5
    assert abs(result.treatment_rate - treated_share) < 1e-9
    assert abs(result.total_treatment_cost - 2 * unit_cost) < 1e-9  # two treated rows
    assert abs(result.incremental_value - treated_share * (effect - unit_cost)) < 1e-9
    assert abs(result.reference_value - 5.0) < 1e-9
    # Randomized probabilities are known, so the value is a point estimate with no interval.
    assert result.uncertainty == "point_only"


if __name__ == "__main__":
    main()
