#!/usr/bin/env python3
"""Assess a fitted model's predictive adequacy, and detect when the world moves away from it.

Question: a model was fitted in one period. How well does it predict held-out cases from the
same period, how much worse does it do in a later period whose feature has shifted, and does
the drift diagnostic see the shift on the feature that moved but not on one that did not?

Data. Simulated from a declared mechanism: ``y = x + 0.5 x^2 + noise`` (noise sd 0.5). The
training period draws ``x ~ N(0, 1)``; the later period draws ``x ~ N(1.5, 1)``. A control
feature ``c ~ N(0, 1)`` is unchanged in both periods. The model is a straight line fitted by
least squares on the training rows, with a Gaussian predictive law whose sd is the training
residual sd; the true curvature is deliberately missing from it.

What the assertions check. The assessment's CRPS equals the Gaussian closed form computed in
numpy. Rows used for fitting are not the rows assessed, so the overlap label is
``held_out_verified``; scoring the training rows with ``require_held_out`` is refused. The
later period scores worse than the held-out cases of the training period. The PIT-based
drift of the shifted feature is large and that of the control feature is near zero.

What it does NOT claim. A score is not causal validity, and the drift value has no
significance level: it is a labelled diagnostic, not a test.
"""

from __future__ import annotations

# %%
import math

import numpy as np
from antecedent import predictive_assessment as pa

N_TRAIN, N_HELD, N_LATER = 600, 400, 400
NOISE_SD = 0.5
SHIFT = 1.5
TARGET = pa.Target("outcome", "customers", "units")
rng = np.random.default_rng(31)


def phi_cdf(z: np.ndarray) -> np.ndarray:
    """The standard normal distribution function."""
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z])


def crps_gaussian(mean: np.ndarray | float, sd: float, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mean, sd^2) at each outcome."""
    z = (y - mean) / sd
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return sd * (z * (2.0 * phi_cdf(z) - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def draw(n: int, x_mean: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(x_mean, 1.0, size=n)
    c = rng.normal(0.0, 1.0, size=n)
    y = x + 0.5 * x**2 + NOISE_SD * rng.normal(size=n)
    return x, c, y


# %% [markdown]
# ## Simulate the training period, a held-out set and a later period
#
# The training rows fit the model. The held-out rows come from the same period; the later
# rows come from a period in which the feature ``x`` has moved up by ``SHIFT``.

# %%
x_tr, c_tr, y_tr = draw(N_TRAIN, 0.0)
x_ho, c_ho, y_ho = draw(N_HELD, 0.0)
x_lt, c_lt, y_lt = draw(N_LATER, SHIFT)
print(f"Rows: {N_TRAIN} train, {N_HELD} held out, {N_LATER} later (x shifted by {SHIFT})")


# %% [markdown]
# ## Fit a straight line
#
# The model is a deliberate simplification: a line cannot follow the curvature, which matters
# little where the training data lie and much further out.

# %%
slope, intercept = np.polyfit(x_tr, y_tr, 1)
fit_sd = float(np.std(y_tr - (intercept + slope * x_tr), ddof=2))
print(f"Fitted line: y = {intercept:.2f} + {slope:.2f} x; residual sd {fit_sd:.2f}")


def predictor(name: str, x: np.ndarray, fit: pa.FitDeclaration) -> pa.Predictor:
    laws = [pa.Gaussian(float(intercept + slope * v), fit_sd) for v in x]
    return pa.Predictor(name, laws, TARGET, fit=fit)


train_keys = pa.keys("train", range(N_TRAIN))
fit = pa.FitDeclaration.fitted_on(train_keys)


# %% [markdown]
# ## Assess held-out cases of the training period
#
# The observations are identified by row, and the fit declaration lists the rows the model
# was fitted on, so the assessment can verify that no assessed row was used for fitting.

# %%
held_keys = pa.keys("heldout", range(N_HELD))
held = pa.assess(
    pa.Observations(TARGET, held_keys, "real", y_ho),
    [predictor("line", x_ho, fit)],
    pa.ValidationDesign.held_out_split(),
    scores=("crps", "log_score"),
    dependence=pa.Dependence.iid(),
    require_held_out=True,
)
entry = held.report.predictor("line")
crps_held = entry.score("crps").mean
expected = float(np.mean(crps_gaussian(intercept + slope * x_ho, fit_sd, y_ho)))
print(f"Held-out CRPS {crps_held:.4f} (numpy closed form {expected:.4f})")
print(f"Overlap: {entry.overlap.label}; log score {entry.score('log_score').mean:.4f}")
assert abs(crps_held - expected) < 1e-8
assert entry.overlap.label == "held_out_verified"
print("Diagnostics:", entry.diagnostics.label, "| calibration", entry.diagnostics.calibration)


# %% [markdown]
# ## Assess the later period
#
# Same model, same scores, later cases. The score is worse because the line extrapolates
# poorly where the curvature matters.

# %%
later_keys = pa.keys("later", range(N_LATER))
later = pa.assess(
    pa.Observations(TARGET, later_keys, "real", y_lt),
    [predictor("line", x_lt, fit)],
    pa.ValidationDesign.held_out_split(),
    scores=("crps", "log_score"),
    dependence=pa.Dependence.iid(),
    require_held_out=True,
)
crps_later = later.report.predictor("line").score("crps").mean
print(f"Later-period CRPS {crps_later:.4f} versus held-out {crps_held:.4f}")
for cov in later.report.predictor("line").diagnostics.coverage or ():
    print(f"  central {cov.level:.0%} interval covers {cov.empirical_coverage:.2f} of cases")
assert crps_later > 1.3 * crps_held


# %% [markdown]
# ## Detect drift on the feature that moved
#
# The drift route compares probability-integral-transform values of two windows. Here each
# feature is passed through the training marginal ``N(0, 1)``: if the window still follows
# it, the values are uniform. The shifted feature drifts; the control feature does not.

# %%
drift_x = pa.distribution_drift(phi_cdf(x_tr), phi_cdf(x_lt), bins=10)
drift_c = pa.distribution_drift(phi_cdf(c_tr), phi_cdf(c_lt), bins=10)
for label, d in (("Shifted feature x", drift_x), ("Control feature c", drift_c)):
    print(f"{label}: KS {d.ks_distance:.3f}, PSI {d.population_stability_index:.3f}")
print(f"Labelled {drift_x.label}, calibration {drift_x.calibration} (no significance level)")
# Independent reference for the KS distance of the shifted feature.
grid = np.sort(np.concatenate([phi_cdf(x_tr), phi_cdf(x_lt)]))
ref = np.searchsorted(np.sort(phi_cdf(x_tr)), grid, side="right") / N_TRAIN
cmp_ = np.searchsorted(np.sort(phi_cdf(x_lt)), grid, side="right") / N_LATER
assert abs(drift_x.ks_distance - float(np.max(np.abs(ref - cmp_)))) < 1e-9
assert drift_x.ks_distance > 0.4 and drift_x.population_stability_index > 0.5
assert drift_c.ks_distance < 0.15 and drift_c.population_stability_index < 0.1


# %% [markdown]
# ## A typed refusal: assessing the rows the model was fitted on
#
# Scoring the training rows and insisting on held-out evidence is refused, never relabelled.

# %%
try:
    pa.assess(
        pa.Observations(TARGET, train_keys, "real", y_tr),
        [predictor("line", x_tr, fit)],
        pa.ValidationDesign.held_out_split(),
        scores=("crps",),
        dependence=pa.Dependence.iid(),
        require_held_out=True,
    )
except pa.PredictiveAssessmentRefusal as refusal:
    print("Refused:", refusal.reason_code, refusal.detail)
    assert refusal.reason_code == "evidence_double_counted"
else:
    raise AssertionError("an in-sample predictor must be refused when held-out is required")
