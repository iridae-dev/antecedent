#!/usr/bin/env python3
"""Thresholds and instruments: sharp and fuzzy discontinuity, a regression kink and IV bounds.

The question: what does a treatment do when a rule at a threshold, or a randomized encouragement,
moves who gets it? Each design below runs on data simulated from a declared mechanism with a
known truth: a sharp jump of 2 at a cutoff; a fuzzy design whose cutoff raises the chance of
treatment from 0.1 to 0.9 and whose treatment effect is 3; a kink where a dose changes slope at
the cutoff and the dose effect is 3; and a binary instrument with 50% compliers, 20%
always-takers and 30% never-takers, an average effect of 0.25 and a complier effect of 0.40.

The assertions compare the sharp, fuzzy and kink estimates with those known values within a
stated tolerance, check that the average effect lies inside the identified IV set, and check
the Wald ratio against the complier effect. Three requests are refused with their reasons.

Discontinuity answers are local to the cutoff (and, when fuzzy, to the units the cutoff moves).
IV bounds are an identified set, never a point; the instrument's independence and exclusion are
declared, not tested. The fuzzy and kink results publish a pointwise 95% interval with their
support and calibration status, which the script prints but does not assert."""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent.errors import CausalError
from antecedent.quasi import (
    FuzzyRegressionDiscontinuity,
    IvBounds,
    RegressionKink,
    SharpRegressionDiscontinuity,
)

SEED = 20260612
N_RD = 30_000
JUMP, FUZZY_EFFECT, KINK_EFFECT = 2.0, 3.0, 3.0
TOL_SHARP, TOL_FUZZY, TOL_KINK = 0.2, 0.4, 0.6
TYPE_SHARE = np.array([0.5, 0.2, 0.3])  # complier, always-taker, never-taker
Y0_PROB = np.array([0.3, 0.4, 0.2])
Y1_PROB = np.array([0.7, 0.5, 0.3])


def refused(call) -> CausalError:
    """Run `call` and return the refusal it raises; anything else is a failure."""
    try:
        call()
    except CausalError as error:
        return error
    raise AssertionError("the call should have been refused")


def show_refusal(label: str, error: CausalError) -> None:
    code = getattr(error, "reason_code", None)
    first = (str(error).strip().splitlines() or [""])[0][:160]
    print(f"{label}: {type(error).__name__} " + (f"reason_code={code}" if code else f"- {first}"))


rng = np.random.default_rng(SEED)
running = rng.uniform(-2.0, 2.0, size=N_RD)
above = running >= 0.0

# %% [markdown]
# ## Sharp discontinuity
#
# Treatment is exactly the rule `running >= 0`. The effect is the jump of the outcome at the
# cutoff, using units inside the bandwidth. Look at the jump against the simulated 2.

# %%
treated = above.astype(float)
sharp_outcome = 1.0 + 0.3 * running + JUMP * treated + 0.3 * rng.normal(size=N_RD)
sharp_data = {"t": treated, "y": sharp_outcome, "r": running}
sharp_query = SharpRegressionDiscontinuity("y", "t", "r", cutoff=0.0, bandwidth=1.5)
sharp_result = ant.analyze(sharp_data, query=sharp_query, execution=ant.Execution(seed=3))
sharp = sharp_result.value
print(
    f"sharp jump {sharp.answer.value:.3f}  (simulated {JUMP}), se {sharp.estimate.se_analytic:.3f}"
)
print("calibration:", sharp_result.inspect().calibration.status)
assert abs(sharp.answer.value - JUMP) < TOL_SHARP, sharp.answer.value

# %% [markdown]
# ## Fuzzy discontinuity
#
# Crossing the cutoff only shifts the chance of treatment (0.1 below, 0.9 above). The fuzzy
# estimate is the outcome jump divided by the treatment jump, so it is the effect for units the
# cutoff moves. Compare the pieces: reduced form about 2.4, first stage about 0.8, ratio 3.

# %%
take_up = np.where(above, 0.9, 0.1)
fuzzy_treatment = (rng.random(N_RD) < take_up).astype(float)
fuzzy_outcome = (
    1.0
    + 0.4 * running
    + 0.2 * running**2
    + FUZZY_EFFECT * fuzzy_treatment
    + 0.3 * rng.normal(size=N_RD)
)
fuzzy_data = {"t": fuzzy_treatment, "y": fuzzy_outcome, "r": running}
fuzzy_result = ant.analyze(
    fuzzy_data, query=FuzzyRegressionDiscontinuity("y", "t", "r", cutoff=0.0, bandwidth=1.5)
)
fuzzy = fuzzy_result.value.local_polynomial_ratio
print(f"fuzzy effect {fuzzy.estimate:.3f}  (simulated {FUZZY_EFFECT})")
print(
    f"reduced form {fuzzy.reduced_form_discontinuity:.3f}, "
    f"first stage {fuzzy.first_stage_discontinuity:.3f}"
)
print(f"window counts left {fuzzy.observations_left}, right {fuzzy.observations_right}")
print("uncertainty:", fuzzy.uncertainty, "| support:", fuzzy.support_status)
print("calibration:", fuzzy_result.inspect().calibration.status)
assert abs(fuzzy.estimate - FUZZY_EFFECT) < TOL_FUZZY, fuzzy.estimate
assert abs(fuzzy.first_stage_discontinuity - 0.8) < 0.1

# %% [markdown]
# ## Regression kink
#
# Here the dose is continuous and only its slope changes at the cutoff (by 0.8). The kink
# estimate divides the outcome's slope change by the dose's slope change. Noise is kept low
# because slope changes are estimated less precisely than jumps.

# %%
dose = 1.0 + 0.2 * running + 0.8 * np.maximum(running, 0.0) + 0.1 * rng.normal(size=N_RD)
kink_outcome = (
    2.0 + 1.5 * running + 0.5 * running**2 + KINK_EFFECT * dose + 0.1 * rng.normal(size=N_RD)
)
kink_result = ant.analyze(
    {"dose": dose, "y": kink_outcome, "r": running},
    query=RegressionKink("y", "dose", "r", cutoff=0.0, bandwidth=1.5),
)
kink = kink_result.value.local_polynomial_ratio
print(f"kink effect {kink.estimate:.3f}  (simulated {KINK_EFFECT})")
print(f"dose slope change {kink.first_stage_discontinuity:.3f}  (simulated 0.8)")
print("calibration:", kink_result.inspect().calibration.status)
assert kink.kink is True
assert abs(kink.estimate - KINK_EFFECT) < TOL_KINK, kink.estimate

# %% [markdown]
# ## IV bounds under a binary instrument
#
# A coin-flip instrument moves treatment for compliers only. A binary instrument alone does not
# pin down the average effect, so the answer is a set of values consistent with the observed
# law. The simulated average effect is 0.25. The Wald ratio targets the complier effect, 0.40.

# %%
n_iv = 20_000
kind = rng.choice(3, size=n_iv, p=TYPE_SHARE)
z = rng.integers(0, 2, size=n_iv)
d = ((kind == 1) | ((kind == 0) & (z == 1))).astype(float)
u = rng.random(n_iv)
y = np.where(d == 1, u < Y1_PROB[kind], u < Y0_PROB[kind]).astype(float)

true_ate = float(TYPE_SHARE @ (Y1_PROB - Y0_PROB))
true_late = float(Y1_PROB[0] - Y0_PROB[0])
iv_result = ant.analyze(
    {"z": z.astype(float), "d": d, "y": y},
    query=IvBounds(outcome="y", treatment="d", instrument="z"),
    refute="none",
    inference=ant.Frequentist(bootstrap=100),
    execution=ant.Execution(seed=5),
)
bounds = iv_result.value.iv_bounds
print(f"bounds [{bounds.lower:.3f}, {bounds.upper:.3f}] around the average effect {true_ate:.2f}")
print("answer kind:", iv_result.value.answer.kind)
interval = bounds.interval
print(f"{interval.method}: [{interval.lower:.3f}, {interval.upper:.3f}]")
print("supported:", bounds.supported, "| calibration:", bounds.calibration)

wald = (y[z == 1].mean() - y[z == 0].mean()) / (d[z == 1].mean() - d[z == 0].mean())
print(f"Wald ratio {wald:.3f}  (complier effect {true_late:.2f})")

assert bounds.lower <= true_ate <= bounds.upper
assert bounds.lower < bounds.upper and iv_result.value.answer.kind == "bounds"
assert interval.lower <= bounds.lower and bounds.upper <= interval.upper
assert bounds.supported is False and bounds.calibration == "not_measured"
assert abs(wald - true_late) < 0.1
# The observed cells are the empirical law P(Y=y, D=d | Z=z), arm-major, within an arm in the
# order (y,d) = (0,0) (1,0) (0,1) (1,1); recompute them directly from the sample.
reference = [
    float(np.mean((y[z == arm] == yy) & (d[z == arm] == dd)))
    for arm in (0, 1)
    for dd in (0, 1)
    for yy in (0, 1)
]
assert np.allclose(bounds.cells, reference, atol=1e-9), (bounds.cells, reference)

# %% [markdown]
# ## Where the designs refuse
#
# Sharp discontinuity refuses a treatment column that is not the threshold rule, the fuzzy
# ratio refuses a first stage it cannot separate from zero, and IV bounds refuse columns that
# are not coded 0 and 1.

# %%
not_a_rule = refused(lambda: ant.analyze({**sharp_data, "t": fuzzy_treatment}, query=sharp_query))
show_refusal("sharp design, imperfect compliance", not_a_rule)

flat_treatment = (rng.random(N_RD) < 0.5).astype(float)
weak = refused(
    lambda: ant.analyze(
        {"t": flat_treatment, "y": fuzzy_outcome, "r": running},
        query=FuzzyRegressionDiscontinuity("y", "t", "r", cutoff=0.0, bandwidth=1.5),
    )
)
show_refusal("fuzzy design, no first stage", weak)

miscoded = refused(
    lambda: ant.analyze(
        {"z": z.astype(float), "d": 2.0 * d, "y": y},
        query=IvBounds(outcome="y", treatment="d", instrument="z"),
        refute="none",
    )
)
show_refusal("IV bounds, treatment coded 0/2", miscoded)
assert "threshold rule" in str(not_a_rule) and "coded exactly 0 and 1" in str(miscoded)
