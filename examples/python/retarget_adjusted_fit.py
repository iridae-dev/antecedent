#!/usr/bin/env python3
"""Move a fitted adjusted effect to a different population without refitting.

Question: a logistic outcome model, fitted once on a study sample, gives the average effect of a
treatment on the chance of a good outcome in that sample. How large is the same effect in a
population that has more of the units with a high value of the adjustment covariate z, and can the
answer come from the fit already made?

Data: simulated with a declared mechanism. z ~ N(0, 1); treatment t is Bernoulli with
logit = 0.8 z; the outcome y is Bernoulli with logit = -0.5 + 1.0 t + 0.9 z. The target population
is declared by row weights exp(0.8 z), which reweights the sample toward N(0.8, 1) in z.

What the asserts check: the retargeted effect equals a weighted standardized effect computed here
from an independent logistic fit (numpy Newton steps); it differs from the all-observed effect;
it is close to the effect of the declared mechanism averaged over the target population; the
retargeted result carries a target-weights identity and the same data snapshot as the original,
so nothing was re-collected or refit; and weights that depend on the treatment are refused.

What the result does not claim: the retargeted interval is conditional on the target weights
being fixed. It does not account for estimating the weights, does not propagate bootstrap
uncertainty and is not doubly robust. Its calibration status is printed as reported.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent.outcomes import require_success

SEED = 20
N = 3000
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
TILT = 0.8


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logistic_fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Maximum-likelihood logistic coefficients by Newton steps."""
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        p = expit(design @ beta)
        step = np.linalg.solve(design.T @ (design * (p * (1 - p))[:, None]), design.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def standardized_effect(beta: np.ndarray, z: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean of P(y = 1 | t = 1, z) - P(y = 1 | t = 0, z)."""
    ones, zeros = np.ones_like(z), np.zeros_like(z)
    p1 = expit(beta[0] + beta[1] * ones + beta[2] * z)
    p0 = expit(beta[0] + beta[1] * zeros + beta[2] * z)
    return float(np.sum(weights * (p1 - p0)) / np.sum(weights))


rng = np.random.default_rng(SEED)
z = rng.normal(size=N)
t = (rng.uniform(size=N) < expit(0.8 * z)).astype(float)
y = (rng.uniform(size=N) < expit(-0.5 + 1.0 * t + 0.9 * z)).astype(float)
data = {"z": z, "t": t, "y": y}

# Effect of the declared mechanism in the target population N(0.8, 1), by quadrature.
grid = np.linspace(-6.0, 7.0, 4001)
density = np.exp(-0.5 * (grid - TILT) ** 2)
true_target = float(
    np.sum(density * (expit(0.5 + 0.9 * grid) - expit(-0.5 + 0.9 * grid))) / np.sum(density)
)
print(f"effect of the declared mechanism in the target population: {true_target:.4f}")

# %% [markdown]
# ## Fit once
#
# Fit the logistic model on the sample with no bootstrap. The study keeps the fitted model and the
# complete-case rows, and that retained fit is what a retarget reuses.

# %%
fitted = require_success(
    ant.analyze(
        data,
        graph=GRAPH,
        query=ant.AverageEffect("t", "y"),
        estimator="glm.adjustment",
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        execution=ant.Execution(seed=SEED),
    )
)
design = np.column_stack([np.ones(N), t, z])
beta = logistic_fit(design, y)
all_observed = standardized_effect(beta, z, np.ones(N))
print(f"all-observed effect: {fitted.answer.value:.4f} (independent {all_observed:.4f})")
assert abs(fitted.answer.value - all_observed) < 1e-5

# %% [markdown]
# ## Retarget to the new population
#
# Declare the target by one weight per row, in the order of the rows, and say which covariates the
# weights depend on. They must be adjustment covariates; weights that depend on the treatment or
# its descendants are refused (below).

# %%
weights = np.exp(TILT * z)
retargeted = require_success(fitted.study.retarget(weights, depends_on=("z",)))
independent = standardized_effect(beta, z, weights)
print(f"retargeted effect: {retargeted.answer.value:.4f} (independent {independent:.4f})")

assert abs(retargeted.answer.value - independent) < 1e-5
assert abs(retargeted.answer.value - all_observed) > 0.005
assert abs(retargeted.answer.value - true_target) < 0.07

# %% [markdown]
# ## Nothing was refit
#
# The retargeted result is bound to the same data snapshot as the fit it came from, and to the
# identity of the weights. A refit or a new sample would change the snapshot.

# %%
before, after = fitted.inspect(), retargeted.inspect()
print("data snapshot unchanged:", before.data_snapshot_id == after.data_snapshot_id)
print("target weights identity:", after.target_weights_id)
print("original has target weights:", before.target_weights_id is not None)
assert before.data_snapshot_id == after.data_snapshot_id
assert before.target_weights_id is None and after.target_weights_id is not None

# %% [markdown]
# ## The interval and its calibration status
#
# The standard error is the original coefficient covariance pushed through the weighted averaging,
# with the target weights held fixed. The calibration status is reported as the library records
# it; a changed target has no measured coverage of its own.

# %%
estimate = retargeted.estimate
se = getattr(estimate, "se_analytic", None)
if se is not None:
    print(f"standard error (target weights fixed): {se:.4f}")
    assert se > 0.0
print("calibration:", retargeted.calibration.status, retargeted.calibration.reason)

# %% [markdown]
# ## Weights that depend on the treatment refuse
#
# A target population defined through the treatment would change what the effect means, so the
# weights may not depend on the treatment or anything it causes.

# %%
refused = fitted.study.retarget(np.exp(TILT * t), depends_on=("t",))
assert refused.refusal is not None
print("refused:", refused.refusal.code)
print("remedy:", refused.refusal.remedy)
assert refused.refusal.code == "target_weights_not_supported"
