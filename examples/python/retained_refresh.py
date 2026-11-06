#!/usr/bin/env python3
"""Refresh a retained analysis as new data arrive, without refitting from scratch.

Question: a covariate-adjusted effect estimate is already running; two more batches of rows
arrive and a few earlier rows are corrected. Can the retained summary be updated in place and
still agree with a fit over all the data?

The data are simulated from a declared mechanism: a confounder `z` raises both the chance of
treatment and the outcome, and treatment adds exactly 2.0 to the outcome
(`y = 2 t + 1.5 z + noise`). The study's estimator is linear adjustment, whose result rests on a
least-squares summary (`RetainedRefresh.family_of` says so). The refresh keeps that summary:
an append updates it in place, anything else recomputes it from the retained rows.

Assertions compare the refreshed coefficients with an independent numpy least-squares solve over
exactly the rows that should be held, to 1e-9; compare the first batch with the study's own
estimate; and compare the treatment coefficient with the true 2.0 within a few standard errors.
Retention, spill and reload are shown with their typed refusals. Refreshing does not
re-identify the effect or recheck any assumption, and it carries no interval; it only keeps the
least-squares summary current.
"""

from __future__ import annotations

# %%
import tempfile

import antecedent as ant
import numpy as np
from antecedent.errors import CausalError, CausalStateError
from antecedent.outcomes import require_success
from antecedent.state import (
    RefreshMutation,
    RefreshObservation,
    RetainedRefresh,
    RetentionPolicy,
    SpillRegistry,
)

rng = np.random.default_rng(20261106)
TRUE_EFFECT = 2.0


def simulate(n: int) -> dict[str, np.ndarray]:
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    y = TRUE_EFFECT * t + 1.5 * z + rng.normal(0.0, 1.0, size=n)
    return {"t": t, "z": z, "y": y}


def observations(batch: dict[str, np.ndarray], first_id: int) -> list[RefreshObservation]:
    """One refresh row per unit: (intercept, treatment, confounder) and the outcome."""
    return [
        RefreshObservation(first_id + i, (1.0, float(t), float(z)), float(y))
        for i, (t, z, y) in enumerate(zip(batch["t"], batch["z"], batch["y"], strict=True))
    ]


def ols(rows: list[RefreshObservation]) -> np.ndarray:
    """Independent least squares over the rows that should be held."""
    x = np.array([r.x for r in rows])
    y = np.array([r.y for r in rows])
    return np.linalg.lstsq(x, y, rcond=None)[0]


# %% [markdown]
# ## Prepare the study on the first batch
#
# The prepared study fixes the question and estimator. Its estimator decides which summary a
# refresh keeps; the refresh then starts from the same first-batch rows.

# %%
batch0, batch1, batch2 = simulate(200), simulate(60), simulate(60)
graph = ant.Dag.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
study = ant.prepare(
    batch0,
    graph=graph,
    query=ant.AverageEffect("t", "y"),
    estimator="linear.adjustment.ate",
    refute=False,
)
family = RetainedRefresh.family_of(study)
rows = observations(batch0, 0)
refresh = RetainedRefresh.start(study, source="batch-0", width=3, rows=rows)
print("summary family:", family, "| coefficients:", np.round(refresh.result()["beta"], 4))

assert family == refresh.family == "ols"
assert np.allclose(refresh.result()["beta"], ols(rows), atol=1e-9)
study_ate = require_success(ant.estimate(study)).estimate.ate
coefficient = refresh.result()["beta"][1]
print(f"study estimate={study_ate:.4f} | refresh treatment coefficient={coefficient:.4f}")
assert abs(study_ate - coefficient) < 1e-6
refresh.verify()

# %% [markdown]
# ## Append new batches
#
# An append is applied to the summary in place; the call returns `"updated"`. After each batch the
# cached result is compared with a fresh least-squares solve over every row so far.

# %%
for name, batch, first_id in (("batch-1", batch1, 1000), ("batch-2", batch2, 2000)):
    new_rows = observations(batch, first_id)
    verdict = refresh.apply(RefreshMutation.append("batch-0", new_rows))
    rows = rows + new_rows
    beta = np.array(refresh.result()["beta"])
    print(f"{name}: {verdict} | n={refresh.result()['n']} | effect={beta[1]:.4f}")
    assert verdict == "updated"
    assert np.allclose(beta, ols(rows), atol=1e-9)
    assert refresh.result()["n"] == len(rows)
    refresh.verify()

resid = np.array([r.y for r in rows]) - np.array([r.x for r in rows]) @ beta
x = np.array([r.x for r in rows])
se = np.sqrt(resid.var(ddof=3) * np.linalg.inv(x.T @ x)[1, 1])
print(f"effect={beta[1]:.3f} (truth {TRUE_EFFECT}), standard error {se:.3f}")
assert abs(beta[1] - TRUE_EFFECT) < 4.0 * se

# %% [markdown]
# ## Corrections and windows recompute
#
# Changing a row's weight, masking a row, evicting a row or keeping only a trailing window cannot
# be done by adding to the summary, so each is recomputed from the retained rows and still
# agrees with a fresh solve.

# %%
target = rows[5].id
assert refresh.apply(RefreshMutation.set_mask(target, True)) == "recomputed"
live = [r for r in rows if r.id != target]
assert np.allclose(refresh.result()["beta"], ols(live), atol=1e-9)

assert refresh.apply(RefreshMutation.evict(rows[6].id)) == "recomputed"
live = [r for r in live if r.id != rows[6].id]
assert np.allclose(refresh.result()["beta"], ols(live), atol=1e-9)

assert refresh.apply(RefreshMutation.window(200)) == "recomputed"
live = live[-200:]
assert np.allclose(refresh.result()["beta"], ols(live), atol=1e-9)
refresh.verify()
print("after mask, evict and window:", refresh.result()["rows"], "rows held")

# A refused change leaves the summary exactly as it was.
before = dict(refresh.result().values)
try:
    refresh.apply(RefreshMutation.append("batch-0", [live[-1]]))
except CausalError as refusal:
    print("refused duplicate row:", type(refusal).__name__, "|", str(refusal)[:100])
else:
    raise AssertionError("a duplicate row id should be refused")
assert dict(refresh.result().values) == before

# %% [markdown]
# ## Say what you retain
#
# A summary-only copy keeps the coefficients but not the rows. It gives the same answer and names
# what it can no longer do.

# %%
policy = RetentionPolicy.sufficient_statistics()
print("mode:", policy.result_mode, "| no longer serves:", policy.no_longer_serves)
summary = refresh.with_retention(policy)
assert summary.result()["beta"] == refresh.result()["beta"]
try:
    policy.require("refit_from_retained_rows")
except CausalStateError as refusal:
    print("refused:", refusal)
else:
    raise AssertionError("summary-only retention cannot refit")
try:
    summary.apply(RefreshMutation.evict(live[0].id))
except CausalStateError as refusal:
    print("refused change on a summary:", str(refusal)[:100])
else:
    raise AssertionError("a summary-only refresh cannot recompute")

# %% [markdown]
# ## Spill to disk and reload
#
# A refresh can be written under an admission budget and read back. The reload is an independent
# branch that still has its rows, so changes and verification work on it.

# %%
with tempfile.TemporaryDirectory() as directory:
    registry = SpillRegistry(directory, 1 << 20)
    spilled = refresh.spill(registry, "study")
    print(spilled, "| registry:", registry.stats["spills"], "spill(s)")
    assert spilled.on_disk and spilled.evicted == ()
    try:
        spilled.require("refit_from_retained_rows")
    except CausalStateError as refusal:
        print("refused on the spilled copy:", str(refusal)[:100])
    else:
        raise AssertionError("a spilled summary must be reloaded first")
    back = spilled.reload()
    assert back.result().values == refresh.result().values
    back.verify()
    assert back.apply(RefreshMutation.evict(live[1].id)) == "recomputed"
    assert refresh.result().values == before
print("spilled copy reloaded with the same coefficients")
