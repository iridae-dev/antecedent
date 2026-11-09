#!/usr/bin/env python3
"""How does the outcome change with dose?

Estimate a response curve, then ask three follow-up questions:

- How steep is the curve at one dose? Use a local derivative.
- What percentage change in outcome follows a percentage change in dose? Use elasticity.
- What is the average slope across the observed data? Use an average derivative.

For each answer, check whether the graph identifies it, whether the data support
it, and what uncertainty is reported.

The simulated structural curve is E[outcome | do(dose = d)] = 12 + 3 ln d - 0.3 d^2. The
script checks the adjusted curve at dose 4.5 (within 0.6) and the local slope at 2.5 (within
0.5) against it; the elasticity, the average derivative and the dose beyond the data are
reported, not checked. Pointwise intervals are not a simultaneous band.
"""

from __future__ import annotations

# %%
import numpy as np
from antecedent import (
    AverageDerivative,
    Elasticity,
    PointDerivative,
    ResponseCurve,
    analyze,
)
from antecedent.outcomes import require_success

SEED = 500
N = 1_500

# %% [markdown]
# ## A confounded nonlinear dose process
#
# Baseline severity affects both the dose selected by clinicians and the outcome. The graph
# therefore requires adjustment for `severity`. Dose is positive, which gives the elasticity a
# meaningful treatment scale.

# %%
rng = np.random.default_rng(SEED)
severity = rng.normal(size=N)
dose = np.clip(2.5 + 0.65 * severity + rng.normal(scale=0.65, size=N), 0.25, 5.0)
outcome = (
    12.0 + 3.0 * np.log(dose) - 0.30 * dose**2 + 1.25 * severity + rng.normal(scale=0.45, size=N)
)

data = {"severity": severity, "dose": dose, "outcome": outcome}
graph = [
    ("severity", "dose"),
    ("severity", "outcome"),
    ("dose", "outcome"),
]

print(float(dose.min()), float(np.median(dose)), float(dose.max()))

# %% [markdown]
# ## Ask four questions
#
# Each query takes the treatment name first and the outcome name second.
# For derivatives and elasticities, choose a bandwidth in `estimator_config`;
# Silverman's rule is not supported for those queries.
#
# The curve includes a dose of `5.25`, beyond the data's maximum of `5.0`.
# Watch for the support warning at that point: requesting a dose does not mean
# the data can support an estimate there.

# %%
curve_query = ResponseCurve(
    "dose",
    "outcome",
    grid=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.25],
)
point_query = PointDerivative("dose", "outcome", at=2.5)
elasticity_query = Elasticity("dose", "outcome", at=2.5)
average_query = AverageDerivative("dose", "outcome", weighting="observed")

curve = require_success(analyze(data, graph=graph, query=curve_query))
point = require_success(
    analyze(data, graph=graph, query=point_query, estimator_config={"bandwidth": 0.35})
)
elasticity = require_success(
    analyze(data, graph=graph, query=elasticity_query, estimator_config={"bandwidth": 0.35})
)
average = require_success(analyze(data, graph=graph, query=average_query))

# %% [markdown]
# ### Reuse the analysis and read its report
#
# Keep `result.study` to run the same analysis again. `ant.estimate(study)` runs
# the prepared branch; `ant.rebind(study, data=new_data)` returns a new branch on the
# new data without changing the original. Earlier results stay unchanged.
#
# The report explains the answer and its limitations. Check calibration too:
# `not_measured` means no coverage record measured this construction; the interval is
# reported without a calibration claim.
# A passing diagnostic does not fill that gap.

# %%
study = curve.study
report = curve.inspect().to_dict()
print("Answer:", curve.answer)
print("Calibration:", curve.calibration.status, curve.calibration.reason)
print(report)

# %% [markdown]
# ## Inspect the orthogonal result axes
#
# An identified estimand can have weak or absent empirical support. Likewise, `pointwise`
# uncertainty describes coverage at individual grid points; it is not a simultaneous band for
# the whole curve. The provenance operation identifies the implemented algorithm, while
# assumptions remain inspectable rather than implicit.

# %%
print(curve.inspect().to_dict())

# %%
curve_rows = [
    {"dose": point_values[0], "mean_response": response_values[0]}
    for point_values, response_values in zip(
        curve.response.points, curve.response.values, strict=True
    )
]
print(curve_rows)

# %%
print(
    {
        "local_derivative_at_2_5": point.estimate,
        "elasticity_at_2_5": elasticity.estimate,
        "observed_law_average_derivative": average.estimate,
        "local_support": point.support.status,
        "average_support": average.support.status,
    }
)


# %%
# The simulation above fixes the structural curve: E[outcome | do(dose = d)] =
# 12 + 3 ln d - 0.3 d^2 (severity has mean zero). Confounding by severity biases the
# unadjusted slope at dose 2.5 to about +0.66, far from the structural -0.3, and lifts the
# unadjusted level at dose 4.5 by about 1.9; the adjusted answers must sit near the truth.
def structural(d):
    return 12.0 + 3.0 * np.log(d) - 0.3 * d**2


by_dose = {
    round(point_values[0], 2): response_values[0]
    for point_values, response_values in zip(
        curve.response.points, curve.response.values, strict=True
    )
}
assert abs(by_dose[4.5] - structural(4.5)) < 0.6, by_dose[4.5]
assert abs(point.estimate - (3.0 / 2.5 - 0.6 * 2.5)) < 0.5, point.estimate

# %% [markdown]
# ## Interpretation
#
# The curve describes levels under intervention. The point derivative is the local slope at
# dose 2.5. The elasticity rescales a local derivative and is not a separate robustness check.
# The average derivative instead integrates slopes over the observed treatment law, so it
# answers a population-weighted question.
#
# Do not report the final curve point as empirically supported merely because an estimator
# returned a number. Inspect `curve.support`, its diagnostics, and its warnings. Also do not
# describe pointwise intervals as a simultaneous confidence band.
