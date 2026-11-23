#!/usr/bin/env python3
"""What is the intent-to-treat (ITT) effect of a randomized assignment on revenue?

Data: 400 simulated accounts assigned by a fair coin; assignment raises revenue by 2.0 (unit
noise). The design is declared, so no graph is needed. Its row-aligned inputs (realized
assignment, unit ids) are named by data column here (arrays also work); `prepare` resolves the
names and freezes the design, and `rebind` reuses it with outcome-only data.

The script asserts the ITT estimate is within 0.5 of 2.0. The interval's support and calibration
status are printed as reported, not asserted; interval claims for randomized designs are rows of
docs/graphless-support-matrix.md.

Run: `python examples/python/experiment_randomized_effect.py`."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent import analyze
from antecedent.experiment import ExperimentDesign, RandomizedEffect
from antecedent.interference import BernoulliAssignment


def main() -> None:
    rng = np.random.default_rng(0)
    n = 400
    assigned = rng.random(n) < 0.5
    revenue = 2.0 * assigned.astype(float) + rng.normal(size=n)
    account_id = np.array([f"account-{i}" for i in range(n)])

    # The design states how assignment happened (independent fair coins) and which columns
    # hold the realized assignment and the unit ids; randomization identifies the effect.
    design = ExperimentDesign(
        BernoulliAssignment(0.5),
        realized_assignment="assigned",
        assignment_units="account_id",
        outcome_units="account_id",
    )
    query = RandomizedEffect("revenue", design)
    data = {"revenue": revenue, "assigned": assigned, "account_id": account_id}

    result = analyze(data, query=query)
    fit = result.value.randomized_effect
    print("Calibration:", result.value.calibration.status)
    print(f"ITT effect={fit.effect:.4f} 95% interval={fit.interval_95}")
    print("uncertainty:", fit.uncertainty, "support:", fit.support_status)
    # The simulated assignment lifts revenue by 2.0.
    assert abs(fit.effect - 2.0) < 0.5, fit.effect

    # A new outcome measured on the same assigned units needs no new design: rebinding keeps
    # the frozen assignment and unit ids and takes outcome-only data.
    fresh = 2.0 * assigned.astype(float) + rng.normal(size=n)
    updated = antecedent.estimate(antecedent.rebind(result.study, data={"revenue": fresh}))
    print("After rebinding, effect=", round(updated.value.randomized_effect.effect, 4))


if __name__ == "__main__":
    main()
