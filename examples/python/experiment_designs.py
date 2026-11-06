#!/usr/bin/env python3
"""Analyze common experiment designs: switchback, CUPED, exact randomization, noncompliance.

Question: how do I analyze an experiment whose design is not a plain two-arm coin flip? Four
designs are simulated from declared mechanisms with known truths, each analyzed with the matching
`antecedent.experiment` entry point:

1. A switchback (treatment re-randomized every period within each sequence) with a carryover from
   the previous period. The per-period contrast recovers the direct effect, 2.0.
2. A Bernoulli trial with a strong pre-assignment covariate. CUPED recovers the effect, 2.0, with a
   smaller standard error than the unadjusted contrast.
3. A 12-unit trial analyzed with the exact randomization test; the p-value is checked against an
   enumeration of all 4096 assignments written out in this script.
4. Encouragement with partial uptake. The Wald complier effect recovers 3.0, and an encouragement
   with no first stage is refused.

Assertions compare each headline number with the simulation's parameter within a stated
tolerance (several standard errors) or with a closed form computed here. Standard errors of the
CUPED, ANCOVA-style and switchback utilities are not calibrated coverage claims; the printed
calibration and support fields say what each result does and does not carry. The exact test
covers the sharp null of no effect for any unit and says nothing about effect size, and no result
here speaks to a design other than the one declared.
"""

from __future__ import annotations

# %%
import itertools

import antecedent as ant
import numpy as np
from antecedent.design_models import SwitchbackModel
from antecedent.errors import CausalError
from antecedent.experiment import (
    ComplierEffect,
    ExperimentDesign,
    RandomizedEffect,
    SwitchbackDesign,
    SwitchbackEffect,
    estimate_cuped_effect,
    exact_randomization_test,
)
from antecedent.interference import BernoulliAssignment
from antecedent.outcomes import require_success

rng = np.random.default_rng(20261106)


def bernoulli_design(n: int, assigned: str = "assigned") -> ExperimentDesign:
    """Independent coin-flip assignment; design inputs are named by data column."""
    return ExperimentDesign(
        BernoulliAssignment(0.5),
        realized_assignment=assigned,
        assignment_units="unit",
        outcome_units="unit",
    )


# %% [markdown]
# ## Switchback with carryover
#
# Thirty sequences each re-randomize treatment in 12 periods. The outcome has a direct effect of
# 2.0 and a carryover of 0.8 from the previous period's assignment. Because each period is
# randomized independently, the previous assignment is balanced across the current arms, so the
# per-period contrast still targets the direct effect. The variance clusters by sequence.

# %%
sequences, periods = 30, 12
z = rng.random((sequences, periods)) < 0.5
previous = np.zeros_like(z)
previous[:, 1:] = z[:, :-1]
baseline = rng.normal(0.0, 1.0, size=(sequences, 1))
y = baseline + 2.0 * z + 0.8 * previous + rng.normal(0.0, 0.5, size=(sequences, periods))

seq_ids = [f"s{s}" for s in range(sequences) for _ in range(periods)]
period_ids = [f"p{t}" for _ in range(sequences) for t in range(periods)]
design = SwitchbackDesign(
    z.ravel().tolist(), seq_ids, period_ids, [0.5] * z.size, treatment_arms=("off", "on")
)
query = SwitchbackEffect("y", design)
result = require_success(ant.analyze({"y": y.ravel()}, query=query, refute="none"))
fit = result.randomized_effect
print(f"switchback effect={fit.effect:.3f} se={fit.standard_error:.3f} 95%={fit.interval_95}")
print("uncertainty:", fit.uncertainty, "| support:", fit.support_status)
print("calibration:", result.calibration.status)
assert abs(fit.effect - 2.0) < 4.0 * fit.standard_error, fit.effect
assert fit.interval_95 is not None and fit.interval_95[0] < 2.0 < fit.interval_95[1]

# The no-carryover assumption is recorded on the result rather than silently dropped.
print([a for a in result.assumptions if "carryover" in a])

# %% [markdown]
# ## Declare carryover for a Bayesian switchback
#
# A `SwitchbackModel` must name its carryover and within-sequence dependence. First-order carryover
# is modelled and the direct effect is still near 2.0; unrestricted carryover is refused with a
# typed reason that points to the next attempt.


# %%
def switchback_model(carryover: str) -> SwitchbackModel:
    return SwitchbackModel(
        carryover=carryover,
        dependence="independent",
        effect_precision=0.01,
        nuisance_precision=0.01,
        variance_shape=2.0,
        variance_scale=1.0,
    )


def bayes(model: SwitchbackModel) -> ant.Bayesian:
    return ant.Bayesian(backend="conjugate", n_draws=2000, design_model=model)


bayes_result = require_success(
    ant.analyze(
        {"y": y.ravel()},
        query=query,
        inference=bayes(switchback_model("first_order")),
        execution=ant.Execution(seed=7),
        refute="none",
    )
)
post = bayes_result.posterior
print(f"first-order carryover posterior mean={post.effect_mean:.3f} sd={post.effect_sd:.3f}")
print("calibration:", bayes_result.inspect().calibration)
assert abs(post.effect_mean - 2.0) < 4.0 * post.effect_sd + 0.1, post.effect_mean

try:
    require_success(
        ant.analyze(
            {"y": y.ravel()},
            query=query,
            inference=bayes(switchback_model("unrestricted")),
            refute="none",
        )
    )
except CausalError as refusal:
    first = str(refusal).splitlines()[0][:100]
    print("refused:", type(refusal).__name__, refusal.reason_code, "|", first)
    assert refusal.reason_code == "not_supported"
else:
    raise AssertionError("unrestricted carryover should be refused")

# %% [markdown]
# ## CUPED variance reduction
#
# A pre-assignment covariate that predicts the outcome strongly (coefficient 3.0) is subtracted
# out. The effect is the same 2.0 target; the standard error shrinks because the covariate's
# variation no longer sits in the noise.

# %%
n = 400
assigned = rng.random(n) < 0.5
pre = rng.normal(0.0, 2.0, size=n)
outcome = 2.0 * assigned + 3.0 * pre + rng.normal(0.0, 0.7, size=n)
unit = np.array([f"u{i}" for i in range(n)])
data = {"y": outcome, "assigned": assigned, "unit": unit, "pre": pre}

raw = require_success(
    ant.analyze(data, query=RandomizedEffect("y", bernoulli_design(n)), refute="none")
).randomized_effect
cuped = estimate_cuped_effect(outcome, pre, assigned.tolist(), 0.5)
print(f"unadjusted effect={raw.effect:.3f} se={raw.standard_error:.3f}")
print(
    f"CUPED effect={cuped.effect:.3f} se={cuped.standard_error:.3f} "
    f"coef={cuped.adjustment_coefficient:.3f}"
)
print("CUPED uncertainty:", cuped.uncertainty, "| support:", cuped.support_status)

assert abs(cuped.effect - 2.0) < 4.0 * cuped.standard_error
assert abs(cuped.adjustment_coefficient - 3.0) < 0.2
assert cuped.standard_error < 0.5 * raw.standard_error

# The retained route takes a coefficient fixed outside this trial's outcomes.
fixed = require_success(
    ant.analyze(
        data,
        query=RandomizedEffect(
            "y", bernoulli_design(n), cuped=ant.experiment.FixedCUPED("pre", 3.0)
        ),
        refute="none",
    )
).randomized_effect
print(f"FixedCUPED effect={fixed.effect:.3f} variance bound={fixed.variance_upper_bound:.4f}")
print("uncertainty:", fixed.uncertainty)
assert abs(fixed.effect - 2.0) < 4.0 * np.sqrt(fixed.variance_upper_bound)

# %% [markdown]
# ## Exact randomization test
#
# With 12 units and a fair coin there are 4096 possible assignments. The function enumerates all of
# them and reports the two-sided p-value of the sharp null (no effect for any unit). Here the same
# enumeration is written out with numpy and compared.

# %%
m = 12
small_assigned = np.array([True] * 8 + [False] * 4)
rng.shuffle(small_assigned)
small_y = 1.5 * small_assigned + rng.normal(0.0, 1.0, size=m)

test = exact_randomization_test(small_y, small_assigned.tolist(), 0.5)
print(f"observed={test.observed_effect:.4f} p={test.exact_two_sided_p_value:.5f}")
print("null:", test.null, "| enumerated:", test.assignments_enumerated)

# Independent enumeration: Horvitz-Thompson contrast under p = 0.5 for every assignment.
all_z = np.array(list(itertools.product([0.0, 1.0], repeat=m)))
contrast = (all_z * small_y / 0.5 - (1.0 - all_z) * small_y / 0.5).mean(axis=1)
observed = (small_assigned * small_y / 0.5 - (~small_assigned) * small_y / 0.5).mean()
reference_p = np.mean(np.abs(contrast) >= abs(observed) - 1e-9)
assert test.assignments_enumerated == 2**m == len(all_z)
assert abs(test.observed_effect - observed) < 1e-9
assert abs(test.exact_two_sided_p_value - reference_p) < 1e-12

try:
    exact_randomization_test([1.0] * 21, [True] * 21, 0.5)
except ValueError as refusal:
    print("refused (21 units):", refusal)

# %% [markdown]
# ## Encouragement with partial uptake
#
# Half the units are encouraged. Encouragement raises uptake of the treatment from 10% to 70%
# (compliers are 60%) and treatment raises the outcome by 3.0, so the complier effect is 3.0
# under the exclusion and monotonicity assumptions that the call declares. When encouragement
# does not change uptake at all, there are no compliers and the effect is refused.

# %%
n = 1500
encouraged = rng.random(n) < 0.5
complier = rng.random(n) < 0.6
always = ~complier & (rng.random(n) < 0.1 / 0.4)
received = np.where(encouraged, complier | always, always)
outcome = 1.0 + 3.0 * received + rng.normal(0.0, 1.0, size=n)
unit = np.array([f"u{i}" for i in range(n)])
data = {"y": outcome, "assigned": encouraged, "unit": unit, "received": received}

cace = require_success(
    ant.analyze(
        data,
        query=ComplierEffect("y", bernoulli_design(n), "received"),
        refute="none",
    )
).randomized_effect
print(f"CACE={cace.effect:.3f} se={cace.standard_error:.3f} interval={cace.interval_95}")
print("estimand:", cace.estimand, "| support:", cace.support_status)
assert abs(cace.effect - 3.0) < 4.0 * cace.standard_error + 0.05, cace.effect

no_uptake_change = {**data, "received": np.zeros(n, dtype=bool)}
try:
    require_success(
        ant.analyze(
            no_uptake_change,
            query=ComplierEffect("y", bernoulli_design(n), "received"),
            refute="none",
        )
    )
except CausalError as refusal:
    code = getattr(refusal, "reason_code", None)
    print("refused:", type(refusal).__name__, code or "", "|", str(refusal).splitlines()[0][:110])
else:
    raise AssertionError("a zero first stage should be refused")
