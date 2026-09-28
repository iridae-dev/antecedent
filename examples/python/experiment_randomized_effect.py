#!/usr/bin/env python3
"""Estimate a randomized experiment's ITT effect, using column-name inputs.

`antecedent.experiment` carries a randomized design through the ordinary
`analyze` / `prepare` API as a retained study. The row-aligned design inputs
(realized assignment and unit ids) may be passed inline as arrays or, as shown
here, named by the data column that holds them; `prepare` resolves those names
and freezes the design. `refresh` then reuses that frozen design with
outcome-only data. Interval claims for this family are rows of the graphless
support matrix (docs/graphless-support-matrix.md).

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

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

    # Row-aligned design inputs are named by data column and resolved at prepare.
    design = ExperimentDesign(
        BernoulliAssignment(0.5),
        realized_assignment="assigned",
        assignment_units="account_id",
        outcome_units="account_id",
    )
    query = RandomizedEffect("revenue", design)
    data = {"revenue": revenue, "assigned": assigned, "account_id": account_id}

    result = analyze(data, query=query)
    fit = result.randomized_effect
    print("Calibration:", result.calibration.status)
    print(f"ITT effect={fit.effect:.4f} 95% interval={fit.interval_95}")
    print("uncertainty:", fit.uncertainty, "support:", fit.support_status)
    # The assignment lifts revenue by 2.0; the ITT estimate recovers it.
    assert abs(fit.effect - 2.0) < 0.5, fit.effect

    # Refresh reuses the frozen design and accepts outcome-only data.
    fresh = 2.0 * assigned.astype(float) + rng.normal(size=n)
    updated = result.study.refresh({"revenue": fresh})
    print("After refresh, effect=", round(updated.randomized_effect.effect, 4))


if __name__ == "__main__":
    main()
