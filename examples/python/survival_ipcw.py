#!/usr/bin/env python3
"""Survival and cumulative incidence when censoring depends on a covariate (IPCW).

Question: a randomized treatment lowers an event hazard, but a risky subgroup is also censored
sooner, so follow-up thins out for exactly the subjects most likely to fail. A Kaplan-Meier curve
that treats censoring as independent then reads too high. If the probability that each subject is
still uncensored (the censoring survival `G`) is known, can inverse-`G` weighting recover the
survival curve?

The data are simulated: a binary risk marker `z` (half the subjects) scales the event hazard by
exp(1.6 z), treatment multiplies it by 0.6, and censoring has rate 0.3 for `z = 1` and 0.01 for
`z = 0`, with everyone administratively censored at tau = 10. Because the simulation draws
censoring from those rates, `G` is exact here: `G_i(t) = exp(-rate_i t)`. The truth is closed-form
(the average over `z` of an exponential survival curve), so every estimate is checked against it.

The example does four things: estimate survival and RMST with weights, show the naive
independent-censoring curve (the same estimator with `G = 1`) is biased, estimate a cause-specific
cumulative incidence with a competing event, and show the refusal of a `G` below the positivity
floor. In real data `G` must be fitted and checked by you; here it is supplied and never fitted
or verified, the results are point-only, and nothing is claimed about delayed entry or intervals.
"""

from __future__ import annotations

# %%
import numpy as np
from antecedent.errors import CausalValueError
from antecedent.survival import (
    CompetingRisksOutcome,
    SurvivalOutcome,
    estimate_cumulative_incidence_ipcw,
    estimate_survival_ipcw,
)

rng = np.random.default_rng(20261106)
TAU = 10.0
BASE_HAZARD, RISK_LOG_HR, TREATMENT_HR = 0.06, 1.6, 0.6
COMPETING_HAZARD = 0.04
CENSOR_RATE = {0: 0.01, 1: 0.3}
MILD_CENSOR_RATE = {0: 0.01, 1: 0.12}


def hazard(z, treated):
    return BASE_HAZARD * np.exp(RISK_LOG_HR * z) * np.where(treated, TREATMENT_HR, 1.0)


def true_survival(t: float, treated: bool) -> float:
    """Marginal survival at t: the average over the risk marker of exp(-h t)."""
    return float(np.mean([np.exp(-hazard(z, treated) * t) for z in (0, 1)]))


def true_rmst(treated: bool) -> float:
    rates = [hazard(z, treated) for z in (0, 1)]
    return float(np.mean([(1 - np.exp(-h * TAU)) / h for h in rates]))


def true_incidence(treated: bool) -> float:
    """Cause-1 cumulative incidence at tau with a competing cause at a constant hazard."""
    out = []
    for z in (0, 1):
        h1 = hazard(z, treated)
        total = h1 + COMPETING_HAZARD
        out.append(h1 / total * (1 - np.exp(-total * TAU)))
    return float(np.mean(out))


def simulate(n: int, competing: bool = False, rates=CENSOR_RATE) -> dict[str, np.ndarray]:
    z = (rng.random(n) < 0.5).astype(int)
    treated = rng.random(n) < 0.5
    t1 = rng.exponential(1.0 / hazard(z, treated))
    rate = np.where(z == 1, rates[1], rates[0])
    censor = rng.exponential(1.0 / rate)
    if competing:
        t2 = rng.exponential(1.0 / COMPETING_HAZARD, size=n)
        failure = np.minimum(t1, t2)
        cause = np.where(t1 <= t2, 1, 2)
    else:
        failure, cause = t1, np.ones(n, dtype=int)
    duration = np.minimum.reduce([failure, censor, np.full(n, TAU)])
    observed = failure <= np.minimum(censor, TAU)
    return {
        "duration": duration,
        "event": observed.astype(float),
        "cause": np.where(observed, cause, 0),
        "treated": treated.astype(float),
        "z": z,
        "rate": rate,
    }


def censoring_survival(data: dict[str, np.ndarray], times: np.ndarray) -> np.ndarray:
    """The exact subject-specific G(t) of the simulation, one row per subject."""
    return np.exp(-data["rate"][:, None] * times[None, :])


def grid(data: dict[str, np.ndarray]) -> np.ndarray:
    """Strictly increasing from zero, containing every observed duration."""
    return np.unique(np.concatenate([[0.0], data["duration"]]))


# %% [markdown]
# ## Simulate the study
#
# 2,500 subjects, individually randomized. The risk marker is not a confounder (treatment is
# randomized); it matters only because it drives censoring.

# %%
data = simulate(2500)
times = grid(data)
censored_share = {z: 1 - data["event"][data["z"] == z].mean() for z in (0, 1)}
print(f"{len(times)} grid times; share not observed to fail: {censored_share}")
query = SurvivalOutcome("duration", "event", "treated", TAU, randomized=True)

# %% [markdown]
# ## Weight by the censoring survival
#
# Each subject's row of `G` gives the probability they were still uncensored at every grid time.
# The weighted Kaplan-Meier curve and RMST are compared with the closed-form truth at the grid time
# nearest 6.

# %%
weighted = estimate_survival_ipcw(
    data, query, times=times, censoring_survival=censoring_survival(data, times)
)
k = int(np.argmin(np.abs(np.array(weighted.times) - 6.0)))
t_star = weighted.times[k]
truth_c, truth_t = true_survival(t_star, False), true_survival(t_star, True)
truth_rmst_diff = true_rmst(True) - true_rmst(False)
print(f"survival at t={t_star:.2f}:")
print(f"  control {weighted.control_survival[k]:.3f} (truth {truth_c:.3f})")
print(f"  treated {weighted.treated_survival[k]:.3f} (truth {truth_t:.3f})")
print(f"RMST difference {weighted.rmst_difference:.3f} (truth {truth_rmst_diff:.3f})")
print("minimum G:", round(weighted.minimum_censoring_survival, 3))
print("provenance:", weighted.censoring_survival_provenance)
print("uncertainty:", weighted.uncertainty, "| support:", weighted.support_status)

assert abs(weighted.control_survival[k] - truth_c) < 0.07
assert abs(weighted.treated_survival[k] - truth_t) < 0.07
assert abs(weighted.rmst_difference - truth_rmst_diff) < 0.7
assert weighted.uncertainty == "point_only"

# %% [markdown]
# ## The independent-censoring curve is biased
#
# The same estimator with `G = 1` for everyone is the ordinary Kaplan-Meier curve. High-risk
# subjects leave the risk set faster, so the survivors that remain are low-risk and the curve
# reads high.

# %%
naive = estimate_survival_ipcw(
    data, query, times=times, censoring_survival=np.ones((len(data["duration"]), len(times)))
)
truth_control = true_survival(t_star, False)
naive_error = naive.control_survival[k] - truth_control
weighted_error = weighted.control_survival[k] - truth_control
print(f"control survival error at t={t_star:.2f}:")
print(f"  naive {naive_error:+.3f}, IPCW {weighted_error:+.3f}")
assert naive_error > 0.06
assert abs(weighted_error) < 0.6 * naive_error

# %% [markdown]
# ## Cumulative incidence with a competing event
#
# A second cause removes subjects at a constant hazard of 0.04 and is not treated as censoring. The
# cause-specific incidence of cause 1 at tau is compared with its closed form. Censoring here is
# milder (rate 0.12 for `z = 1`) so the weights stay small and the estimate is tight.

# %%
cdata = simulate(3500, competing=True, rates=MILD_CENSOR_RATE)
ctimes = grid(cdata)
competing_query = CompetingRisksOutcome(
    "duration", "cause", "treated", target_cause=1, tau=TAU, randomized=True
)
cif = estimate_cumulative_incidence_ipcw(
    cdata, competing_query, times=ctimes, censoring_survival=censoring_survival(cdata, ctimes)
)
truth_difference = true_incidence(True) - true_incidence(False)
truth_ctl, truth_trt = true_incidence(False), true_incidence(True)
print(f"incidence at tau: control {cif.control_incidence[-1]:.3f} (truth {truth_ctl:.3f})")
print(f"                  treated {cif.treated_incidence[-1]:.3f} (truth {truth_trt:.3f})")
print(f"incidence difference {cif.incidence_difference:.3f} (truth {truth_difference:.3f})")
assert abs(cif.control_incidence[-1] - truth_ctl) < 0.04
assert abs(cif.treated_incidence[-1] - truth_trt) < 0.04
assert abs(cif.incidence_difference - truth_difference) < 0.05

# %% [markdown]
# ## A `G` below the positivity floor is refused
#
# If a subject's supplied probability of remaining uncensored is almost zero, its weight would
# dominate the curve. The estimator refuses it rather than returning a number.

# %%
bad = censoring_survival(data, times)
bad[0, len(times) // 2 :] = 0.001
try:
    estimate_survival_ipcw(data, query, times=times, censoring_survival=bad)
except CausalValueError as refusal:
    code = getattr(refusal, "reason_code", None)
    label = type(refusal).__name__ + (f" [{code}]" if code else "")
    print(f"refused: {label}: {str(refusal).splitlines()[0]}")
    assert "positivity floor" in str(refusal)
else:
    raise AssertionError("a G below the floor should be refused")
