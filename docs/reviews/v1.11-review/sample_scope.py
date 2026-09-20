"""Show whether unusable rows can upgrade an unchanged interval's calibration.

Run with python/.venv/bin/python from the repository checkout.
"""

import antecedent as ant
import numpy as np

rng = np.random.default_rng(42)
t = rng.normal(size=100)
y = 2 * t + rng.normal(size=100)
for total in (100, 500, 1000):
    missing = np.full(total - 100, np.nan)
    result = ant.analyze(
        {"t": np.r_[t, missing], "y": np.r_[y, missing]},
        graph=[("t", "y")],
        query=ant.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    print(
        f"input_rows={total}, complete_rows=100, effect={result.effect}, "
        f"analytic_se={result.estimate.se_analytic}, "
        f"calibration={result.calibration.describe()}"
    )
