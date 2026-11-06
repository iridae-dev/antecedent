#!/usr/bin/env python3
"""Ask how far outcomes that are missing not at random could move an effect.

Question: in a randomized comparison, some outcomes are missing, and whether an outcome goes
missing depends on its own value. The complete-case estimate then misses the truth by an
amount no data can reveal. How far could the effect move if the missing outcomes differ from
the observed ones by a declared shift, and does the decision it supports survive?

Data. Simulated from a declared mechanism with a known truth: treatment is assigned at
random, ``y = TRUE_EFFECT * treatment + noise``, and each outcome is missing with probability
``expit(a_arm + 0.8 * y)``, so larger outcomes go missing more often and more so under
treatment. The full-data outcomes are kept aside, which the analysis never sees, to compute
how far the missing outcomes really sit from the observed ones in each arm.

What the assertions check. The complete-case estimate is below the truth. The grid of effects
over the declared shift matches the closed form ``MAR + delta * (f1 - f0)`` (arm missing
fractions ``f1`` and ``f0``). The simulated truth lies inside the assumption range, and at
the grid shift nearest the true departure the effect is within one grid step of the truth.
The tipping point of a declared decision matches its closed form.

What it does NOT claim. The range is an assumption range under a one-parameter shift model,
not a confidence interval, and the data cannot test the departure. The sampling interval at
each grid point is conditional on that point and its calibration is reported, not assumed.
"""

from __future__ import annotations

# %%
import numpy as np
from antecedent import mnar_sensitivity as mnar

TRUE_EFFECT = 2.0
N = 6000
LOWER, UPPER, GRID_POINTS = -3.0, 3.0, 61
THRESHOLD = 1.5
rng = np.random.default_rng(14)


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# %% [markdown]
# ## Simulate outcomes that go missing because of their own value
#
# Treatment is randomized, so the full-data effect is the truth. Missingness depends on the
# outcome and differs by arm. The analysis receives only the treatment and the outcome with
# gaps.

# %%
treatment = (rng.uniform(size=N) < 0.5).astype(float)
y_full = TRUE_EFFECT * treatment + rng.normal(size=N)
p_missing = expit(np.where(treatment == 1.0, -2.0, -1.5) + 0.8 * y_full)
missing = rng.uniform(size=N) < p_missing
outcome = np.where(missing, np.nan, y_full)
n_t, n_c = int(missing[treatment == 1].sum()), int(missing[treatment == 0].sum())
print(f"Rows {N}; missing {int(missing.sum())} (treated {n_t}, control {n_c})")
truth = y_full[treatment == 1].mean() - y_full[treatment == 0].mean()
print(f"Full-data effect (known only here): {truth:.3f}; declared mechanism value {TRUE_EFFECT}")
assert abs(truth - TRUE_EFFECT) < 0.1


# %% [markdown]
# ## The complete-case estimate
#
# Dropping the missing rows treats each arm's missing outcomes as like its observed ones.
# Because larger outcomes go missing more, it is too low, and more so for the treated arm.

# %%
seen = ~missing
complete_case = y_full[seen & (treatment == 1)].mean() - y_full[seen & (treatment == 0)].mean()
print(f"Complete-case estimate: {complete_case:.3f} (full-data effect {truth:.3f})")
assert truth - complete_case > 0.05


# %% [markdown]
# ## Declare a departure and sweep it
#
# The model shifts each arm's missing outcomes by ``delta`` from that arm's observed mean (the
# same shift in both arms). We declare the range ``[-3, 3]``, which contains 0 (the complete-
# case assumption), and a decision that acts when the effect exceeds ``THRESHOLD``.

# %%
result = mnar.mnar_ate_sensitivity(
    treatment,
    outcome,
    model="delta_pattern_mixture",
    loading="common",
    lower=LOWER,
    upper=UPPER,
    grid_points=GRID_POINTS,
    decisions={"launch": THRESHOLD},
)
f1 = result.treated_missing / result.treated_rows
f0 = result.control_missing / (result.rows - result.treated_rows)
slope = f1 - f0
print(f"Missing fractions: treated {f1:.3f}, control {f0:.3f}")
print(f"MAR effect {result.mar.effect:.3f}; observed-case {result.observed_case_effect:.3f}")
rg = result.assumption_range
print(
    f"Assumption range over delta in [{rg.parameter_lower}, {rg.parameter_upper}]: "
    f"[{rg.effect_lower:.3f}, {rg.effect_upper:.3f}]"
)
assert abs(result.mar.effect - complete_case) < 1e-9
assert abs(result.observed_case_effect - complete_case) < 1e-9
for point in result.grid:
    assert abs(point.effect - (complete_case + point.parameter * slope)) < 1e-9
assert not rg.is_confidence_interval and not rg.empirically_established


# %% [markdown]
# ## Does the truth lie in the range at the true departure?
#
# From the kept full data we compute how far the missing outcomes sit above the observed ones
# in each arm. A single common shift that reproduces the full-data effect is
# ``(f1*d1 - f0*d0) / (f1 - f0)``; the grid point nearest to it should land on the truth.

# %%
d1 = y_full[missing & (treatment == 1)].mean() - y_full[seen & (treatment == 1)].mean()
d0 = y_full[missing & (treatment == 0)].mean() - y_full[seen & (treatment == 0)].mean()
delta_true = (f1 * d1 - f0 * d0) / slope
nearest = min(result.grid, key=lambda p: abs(p.parameter - delta_true))
step = (UPPER - LOWER) / (GRID_POINTS - 1)
print(
    f"Missing outcomes sit {d1:.2f} (treated) and {d0:.2f} (control) above observed; "
    f"equivalent common shift {delta_true:.2f}"
)
print(
    f"Effect at the nearest grid shift {nearest.parameter:.1f}: {nearest.effect:.3f} "
    f"(truth {truth:.3f})"
)
assert rg.effect_lower <= truth <= rg.effect_upper
assert abs(nearest.effect - truth) <= 0.5 * step * abs(slope) + 1e-9


# %% [markdown]
# ## Tipping point of the decision
#
# The decision acts when the effect exceeds ``THRESHOLD``. The tipping point is the first
# departure at which that conclusion changes; with a linear shift model it has a closed form.

# %%
launch = result.decisions[0]
print(
    f"Launch acts at MAR: {launch.acts_at_mar}; changes within range: {launch.changes_within_range}"
)
side = launch.tipping.negative if slope > 0 else launch.tipping.positive
expected = (THRESHOLD - result.mar.effect) / slope
print(
    f"Tipping shift {None if side is None else round(side.parameter, 3)} "
    f"(closed form {expected:.3f})"
)
assert launch.acts_at_mar and side is not None
assert abs(side.parameter - expected) < 1e-6
print(
    "Sign of the effect changes within range:",
    result.tipping_point("sign").negative is not None
    or result.tipping_point("sign").positive is not None,
)


# %% [markdown]
# ## Sampling interval and its status
#
# Each grid point also carries a delta-method sampling interval, conditional on that shift. It
# is reported beside the range and is never merged with it.

# %%
mar = result.mar
print(
    f"At MAR: effect {mar.effect:.3f}, sampling interval "
    f"[{mar.sampling.lower:.3f}, {mar.sampling.upper:.3f}]"
)
print(f"Sampling status: {result.sampling_status}; inference claim: {result.inference_claim}")
print(f"Assumption '{result.assumptions[0].id}' status: {result.assumptions[0].status}")
assert mar.sampling is not None and mar.sampling.lower < mar.effect < mar.sampling.upper
assert result.inference_claim == "assumption_range"


# %% [markdown]
# ## A typed refusal: a range that excludes MAR
#
# The declared range must contain 0, the missing-at-random point; one that does not is
# refused.

# %%
try:
    mnar.mnar_ate_sensitivity(
        treatment, outcome, model="delta_pattern_mixture", lower=0.5, upper=2.0
    )
except mnar.MnarSensitivityRefusal as refusal:
    print("Refused:", refusal.code, refusal.detail)
    assert refusal.detail == "mnar_sensitivity.range_excludes_mar"
else:
    raise AssertionError("a range excluding MAR must be refused")
