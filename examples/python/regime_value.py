"""Evaluate a longitudinal treatment regime by inverse-probability g-formula.

`antecedent.regimes.evaluate_regime_value` estimates the value of a prespecified
static or history-adaptive treatment rule under caller-declared sequential
randomization probabilities. It reweights the observed trajectories that follow
the regime and reports the value, effective sample size, matched fraction, and
maximum weight. It is a point utility (`uncertainty == "not_estimated"`) and
does not verify that the supplied probabilities are correct.

With two randomization times at p=0.5, the always-untreated path (0, 0) has
probability 0.25; its sole observed outcome here is 2.0, so the regime value is
2.0. Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent.regimes import evaluate_regime_value


def main() -> None:
    result = evaluate_regime_value(
        outcomes=[2.0, 0.0, 0.0, 0.0],
        treatment_history=[[0, 0], [0, 1], [1, 0], [1, 1]],
        regime=[False, False],  # the static "never treat" rule
        treatment_probabilities=np.full((4, 2), 0.5),
    )

    print(f"regime value={result.value}")
    print(f"effective sample size={result.effective_sample_size}")
    print(f"matched observed fraction={result.matched_observed_fraction}")
    print(f"maximum weight={result.maximum_weight}")
    print("uncertainty:", result.uncertainty, "support:", result.support_status)
    # The never-treat path's sole observed outcome is 2.0, so the value is 2.0.
    assert abs(result.value - 2.0) < 1e-9, result.value


if __name__ == "__main__":
    main()
