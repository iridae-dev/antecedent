#!/usr/bin/env python3
"""Refute an adjusted estimate, then bound the confounder no refutation can see.

Question: a study adjusts for a measured covariate, but a hidden confounder also drives the
treatment and the outcome. Which checks can the estimate pass, and how far could the hidden
confounder have moved it?

Data. Simulated from a declared mechanism with a known truth: a measured covariate ``x`` and a
hidden ``u`` both feed a continuous treatment ``t`` and the outcome ``y``; the effect of ``t``
on ``y`` is exactly ``TRUE_EFFECT``. The analysis never sees ``u``. A second simulation draws
matched pairs whose assignment depends on a hidden uniform score with a known odds ratio.

What the assertions check. The adjusted estimate is biased upward by ``u``. The target-native
refuters (placebo treatment, random common cause) pass on it, as they should: they cannot
detect a confounder nobody measured. The omitted-variable-bias range, evaluated at the
confounder's true partial R-squared values (computed from the simulation), brackets the
full-adjustment coefficient and the simulated truth. For matched pairs, the Rosenbaum upper
bound at the true hidden-bias odds ratio is checked against a sound design and against a
confounded null.

What it does NOT claim. Passing a refuter is not evidence of no unmeasured confounding.
Every number here is a diagnostic with calibration ``not_measured``; the bounds say what a
confounder of a stated strength would do, not that one of that strength exists.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent import validation as v
from antecedent.errors import CausalError
from antecedent.outcomes import require_success

TRUE_EFFECT = 1.5
N = 2000
rng = np.random.default_rng(20)


def ols(columns: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients with an intercept first."""
    design = np.column_stack([np.ones(len(y)), *columns])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def residual(columns: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), *columns])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def partial_r2(a: np.ndarray, b: np.ndarray, given: list[np.ndarray]) -> float:
    """Squared partial correlation of a and b given the columns in ``given``."""
    return float(np.corrcoef(residual(given, a), residual(given, b))[0, 1] ** 2)


# %% [markdown]
# ## Simulate a study with a hidden confounder
#
# The treatment and outcome share the measured ``x`` and the hidden ``u``. The truth is the
# coefficient ``TRUE_EFFECT`` on ``t``; the data handed to the analysis omit ``u``.

# %%
x = rng.normal(size=N)
u = rng.normal(size=N)
t = 0.8 * x + 0.33 * u + rng.normal(size=N)
y = 1.0 + TRUE_EFFECT * t + 1.0 * x + 1.2 * u + rng.normal(size=N)
data = {"t": t, "y": y, "x": x}

restricted = ols([t, x], y)[1]
full = ols([t, x, u], y)[1]
print(f"Adjusting for x only: {restricted:.3f}; also adjusting for hidden u: {full:.3f}")
print(f"Truth: {TRUE_EFFECT}")
assert abs(full - TRUE_EFFECT) < 0.1
assert restricted - TRUE_EFFECT > 0.2  # the hidden confounder biases the study upward


# %% [markdown]
# ## Estimate with adjustment for the measured covariate
#
# The declared graph has ``x`` as the only confounder. The estimate is the one a
# practitioner would report, and it is the quantity the checks below speak to.

# %%
study = ant.prepare(
    data,
    graph=[("x", "t"), ("x", "y"), ("t", "y")],
    query=ant.AverageEffect("t", "y"),
    inference=ant.Frequentist(bootstrap=0),
    refute="none",
    execution=ant.Execution(seed=3),
)
original = ant.estimate(study)
adjusted = require_success(original).answer.value
print(f"Adjusted estimate: {adjusted:.3f}")
assert abs(adjusted - restricted) < 0.02  # the same OLS slope as the numpy reference


# %% [markdown]
# ## Refute the target
#
# The suite refits the study's own target under declared perturbations. Placebo treatment
# should drive the effect to zero; a random common cause should leave it unchanged. Both pass
# here, and neither could have flagged the hidden ``u``.

# %%
report = require_success(
    v.refute_target(original, checks=v.TargetRefutationSpec(suite="full", replicates=12))
)
passed = {check.refuter: check.passed for check in report.reports}
for check in report.reports:
    print(f"  {check.refuter}: passed={check.passed} (informative={check.informative})")
for refusal in report.refused:
    print(f"  refused {refusal.refuter}: {refusal.reason}")
print("Calibration:", report.calibration)
assert passed["target.placebo_treatment"]
assert passed["target.random_common_cause"]
assert report.calibration == "not_measured"


# %% [markdown]
# ## Bound the hidden confounder
#
# The omitted-variable-bias range moves the estimate toward zero as a confounder explains more
# residual treatment variance. Here ``u``'s true partial R-squared values come from the
# simulation, so the range is evaluated at the real strength.

# %%
r2dz = partial_r2(u, t, [x])
r2yz = partial_r2(u, y, [t, x])
print(f"True strength of u: R2 with treatment {r2dz:.3f}, with outcome {r2yz:.3f}")
sens = v.omitted_variable_bias_from_data(
    data, outcome="y", treatment="t", covariates=["x"], r2yz_extreme=r2yz
)
assert abs(sens.summary["estimate"] - restricted) < 1e-6
assert sens.range is not None
grid = [(p.parameter, p.target) for p in sens.range.tested]
# Closed-form bound: se * sqrt(dof) * sqrt(r2yz * r2dz / (1 - r2dz)), dof = N - 3.
se = sens.summary["standard_error"]
for share, moved in grid:
    bound = se * np.sqrt(N - 3) * np.sqrt(r2yz * share / (1 - share))
    assert abs(moved - (restricted - bound)) < 1e-6, (share, moved)
at_or_above = next((s, m) for s, m in grid if s >= r2dz)
print(f"Robustness value RV_1: {sens.range.break_even:.3f} (calibration {sens.range.calibration})")
print(f"At R2 with treatment {at_or_above[0]} (first probed value at or above the truth):")
print(f"  adjusted estimate {at_or_above[1]:.3f}, reported estimate {restricted:.3f}")
slack = 3 * se  # the full regression's coefficient estimates the truth within sampling error
assert at_or_above[1] <= full + 1e-6 <= restricted
assert at_or_above[1] - slack <= TRUE_EFFECT <= restricted


# %% [markdown]
# ## Matched pairs: Rosenbaum bounds
#
# In each pair a hidden uniform score ``w`` nudges which unit is treated, with odds ratio at
# most ``exp(g)``. We simulate a real effect and a null effect under the same confounding and
# look at the upper-bound p-value of the signed-rank test at the true odds ratio.

# %%
PAIRS, G_TRUE_LOG = 250, 1.0
gamma_true = float(np.exp(G_TRUE_LOG))
gammas = [1.0, 1.5, 2.0, gamma_true, 4.0, 6.0, 8.0]


def pair_differences(effect: float, seed: int) -> np.ndarray:
    gen = np.random.default_rng(seed)
    w1, w2 = gen.uniform(size=PAIRS), gen.uniform(size=PAIRS)
    first_treated = gen.uniform(size=PAIRS) < 1 / (1 + np.exp(-G_TRUE_LOG * (w1 - w2)))
    wt, wc = np.where(first_treated, w1, w2), np.where(first_treated, w2, w1)
    noise = 0.7 * gen.normal(size=PAIRS) - 0.7 * gen.normal(size=PAIRS)
    return effect + 1.5 * (wt - wc) + noise


def p_at(result: v.SensitivityResult, gamma: float) -> float:
    return next(p.target for p in result.range.tested if abs(p.parameter - gamma) < 1e-12)


real = v.rosenbaum_bounds(pair_differences(0.8, 1), gammas=gammas)
null = v.rosenbaum_bounds(pair_differences(0.0, 2), gammas=gammas)
for label, result in (("effect 0.8", real), ("null", null)):
    print(
        f"{label}: p(Gamma=1) = {p_at(result, 1.0):.4f}, p(Gamma={gamma_true:.2f}) = "
        f"{p_at(result, gamma_true):.4f}, break-even Gamma = "
        f"{result.range.break_even_significance}"
    )
assert p_at(real, 1.0) <= p_at(real, gamma_true) < 0.05  # holds up at the true hidden bias
assert p_at(null, 1.0) < 0.05 <= p_at(null, gamma_true)  # a confounded null looks significant
assert null.range.break_even_significance < real.range.break_even_significance


# %% [markdown]
# ## A typed refusal
#
# A sensitivity analysis needs a positive standard error; the API refuses anything else rather
# than returning a number.

# %%
try:
    v.omitted_variable_bias(0.5, -1.0, 100.0)
except CausalError as refusal:
    print(f"Refused ({type(refusal).__name__}): {refusal}")
else:
    raise AssertionError("a negative standard error must be refused")
