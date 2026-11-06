#!/usr/bin/env python3
"""Prepare once, then branch and reuse instead of starting over.

Question: three outcomes share one binary treatment ``t`` and one confounder ``z``. After the
first analysis a later wave of data arrives and one question is dropped. How do you answer each
without rebuilding the analysis, and how do you check what the library kept and what it
recomputed?

Data: simulated from a declared mechanism. ``z ~ N(0, 1)``; ``t`` is Bernoulli with a logistic
dependence on ``z``; ``y1 = 2 t + z + noise``, ``y2 = 0.5 y1 - z`` and ``y3 = -0.5 t + 0.5 z +
noise``. The average effects of ``t`` are therefore 2, 1 and -0.5. The later wave has the same
mechanism with a first effect of 3, so its effects are 3, 1.5 and -0.5.

The assertions check three things. The batch estimates recover the declared effects within a
stated tolerance. A completed batch estimated again, a rebound branch and a one-call
``analyze_many`` on the same data all return the same numbers as a fresh run, so reuse never
changes an answer. The earlier results are unchanged by every branch. The tolerance is a
numerical statement about this seeded simulation, not a coverage claim for the intervals.

What this does not claim: ``Reuse`` names categories of retained and invalidated work. It is a
receipt of what the library kept, not a timing, and the exact categories depend on the route.
Nothing here says the declared graph is true of real data.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent.outcomes import require_success

N = 1500
SEED = 11
TOLERANCE = 0.15
GRAPH = [("z", "t"), ("z", "y1"), ("z", "y2"), ("z", "y3"), ("t", "y1"), ("t", "y2"), ("t", "y3")]
OUTCOMES = ("y1", "y2", "y3")


def simulate(effect_on_y1: float, seed: int, n: int = N) -> dict[str, np.ndarray]:
    """A confounded binary treatment and three outcomes with known average effects."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.6 * z))).astype(float)
    y1 = effect_on_y1 * t + z + rng.normal(scale=0.3, size=n)
    y2 = 0.5 * y1 - z
    y3 = -0.5 * t + 0.5 * z + rng.normal(scale=0.3, size=n)
    return {"z": z, "t": t, "y1": y1, "y2": y2, "y3": y3}


def queries(outcomes=OUTCOMES):
    return [ant.AverageEffect("t", outcome) for outcome in outcomes]


def effects(batch) -> list[float]:
    """The point effect of each member, in query order."""
    return [require_success(member).answer.value for member in batch.members]


COMMON = dict(
    graph=GRAPH,
    estimator="aipw",
    inference=ant.Frequentist(bootstrap=0),
    refute=False,
    execution=ant.Execution(seed=SEED),
)
first_wave = simulate(2.0, seed=1)
second_wave = simulate(3.0, seed=2)


# %% [markdown]
# ## Prepare a batch once
#
# `prepare_many` declares all three questions on one table. Preparing and inspecting fit
# nothing; the first fit happens at `estimate`. The truth is 2, 1 and -0.5.

# %%
study = ant.estimation.prepare_many(first_wave, queries=queries(), **COMMON)
print(study.inspect().stage, "| reuse before any run:", study.reuse)
assert study.inspect().stage == "prepared"

batch = require_success(ant.estimate(study))
base = effects(batch)
print("effects on the first wave:", [round(e, 3) for e in base])
for estimate, truth in zip(base, (2.0, 1.0, -0.5), strict=True):
    assert abs(estimate - truth) < TOLERANCE, (estimate, truth)
for member in batch.members:
    print("calibration:", require_success(member).calibration.status)


# %% [markdown]
# ## What the batch shared
#
# Queries that would fit the same cross-fitted propensity model share one fit, and every
# member records that in its diagnostics. A shared fit equals the fit made alone, so a solo
# `analyze` of the first question must give the same effect.

# %%
notes = [d for d in require_success(batch.member(0)).diagnostics if d.startswith("batch.")]
print(notes[0] if notes else "no shared-design note")
assert any("propensity shared" in note for note in notes), notes

solo = require_success(
    ant.analyze(first_wave, query=queries()[0], **COMMON),
)
assert abs(solo.answer.value - base[0]) < 1e-9, (solo.answer.value, base[0])

# One call does the same as prepare then estimate.
one_call = require_success(ant.estimation.analyze_many(first_wave, queries=queries(), **COMMON))
assert max(abs(a - b) for a, b in zip(effects(one_call), base, strict=True)) < 1e-9


# %% [markdown]
# ## Estimate the completed batch again
#
# Executing the completed batch's study again reuses its retained work. The effects must be
# identical, and the original completion is untouched.

# %%
again = require_success(ant.estimate(batch.study))
assert max(abs(a - b) for a, b in zip(effects(again), base, strict=True)) < 1e-12
assert again.retarget_snapshot() == batch.retarget_snapshot()
print("same row snapshot:", again.retarget_snapshot() == batch.retarget_snapshot())


# %% [markdown]
# ## Rebind onto the later wave
#
# `ant.rebind` returns a new prepared branch on new observations; `branch.reuse` says which
# work it kept and which it invalidated. The new answer is compared with a fresh
# `analyze_many` on the same new data, and the first wave's results do not move.

# %%
branch = ant.rebind(batch.study, data=second_wave)
print("rebind  retained:", branch.reuse.retained, "invalidated:", branch.reuse.invalidated)
assert "fits" in branch.reuse.invalidated

rebound = require_success(ant.estimate(branch))
fresh = require_success(ant.estimation.analyze_many(second_wave, queries=queries(), **COMMON))
print("effects on the second wave:", [round(e, 3) for e in effects(rebound)])
assert max(abs(a - b) for a, b in zip(effects(rebound), effects(fresh), strict=True)) < 1e-9
for estimate, truth in zip(effects(rebound), (3.0, 1.5, -0.5), strict=True):
    assert abs(estimate - truth) < TOLERANCE, (estimate, truth)

assert effects(batch) == base  # the earlier completion is unchanged
snapshot_ids = {
    batch.member(0).inspect().data_snapshot_id,
    rebound.member(0).inspect().data_snapshot_id,
}
assert len(snapshot_ids) == 2  # the two waves are different observations


# %% [markdown]
# ## Reconfigure the questions
#
# `ant.reconfigure` changes a declaration and keeps the owned table. Dropping the first outcome
# leaves two queries, and each must equal the same question in the full batch.

# %%
narrowed = ant.reconfigure(batch.study, queries=queries(("y2", "y3")))
print("reconfigure retained:", narrowed.reuse.retained, "invalidated:", narrowed.reuse.invalidated)
assert "data" in narrowed.reuse.retained
smaller = require_success(ant.estimate(narrowed))
assert len(smaller.members) == 2
assert max(abs(a - b) for a, b in zip(effects(smaller), base[1:], strict=True)) < 1e-9
assert len(batch.members) == 3  # the original batch still has three members


# %% [markdown]
# ## Refusals keep their form across branches
#
# Observations change only through `rebind`: passing `data=` to `reconfigure` is a call error.
# A graph with an unmeasured common cause is a scientific refusal that `analyze` returns.

# %%
try:
    ant.reconfigure(batch.study, data=second_wave)
except ant.errors.CausalValueError as error:
    print("call error:", error)
else:
    raise AssertionError("reconfigure(data=...) should be refused")

confounded = ant.Admg.from_edges(["t", "y1"], [("t", "y1")], [("t", "y1")])
refused = ant.analyze(
    {"t": first_wave["t"], "y1": first_wave["y1"]}, graph=confounded, query=queries()[0]
)
assert not refused.accepted and refused.refusal is not None
print("refused:", refused.refusal.code, "|", refused.refusal.detail)
assert refused.refusal.code == "effect_not_identified"
