#!/usr/bin/env python3
"""Run an analysis inside an application that owns the budget, the clock and the stop button.

Question: an application embeds Antecedent. It needs to watch a run, bound the memory a run
may use, stop a run it no longer wants, and know which kind of build it is talking to. All of
these belong to one ``Execution`` object; none changes the scientific declaration.

Data: simulated from a declared mechanism. ``z ~ N(0, 1)``; ``t`` is Bernoulli with a logistic
dependence on ``z``; ``y = 2 t + z + noise``. The average effect of ``t`` is 2, and the graph
``z -> t``, ``z -> y``, ``t -> y`` identifies it.

The assertions check that the effect is near 2 under a plain execution; that attaching stage
and progress observers and a roomy memory limit do not change the number; that a zero-byte
memory limit is refused as a typed resource error before any fit; that a token cancelled
before the call raises a typed cancellation and returns no claim; and that a token cancelled
from a progress callback during the bootstrap stops that stage early and says so in the result.

What this does not claim: a memory limit bounds the library's admitted native pool, not process
memory. Cancellation takes effect at the next polling boundary, not instantly. Which build you
are running is printed, not asserted.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent.errors import CausalCancelledError, CausalResourceError
from antecedent.outcomes import require_success

N = 400
BOOTSTRAP = 80
TRUE_EFFECT = 2.0
TOLERANCE = 0.2
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def simulate(seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)
    t = (rng.random(N) < 1.0 / (1.0 + np.exp(-0.9 * z))).astype(float)
    y = TRUE_EFFECT * t + z + rng.normal(scale=0.4, size=N)
    return {"z": z, "t": t, "y": y}


data = simulate()
query = ant.AverageEffect("t", "y")


def run(execution: ant.Execution, bootstrap: int = 0):
    """Analyze under ``execution`` and return the unwrapped result."""
    return require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=query,
            inference=ant.Frequentist(bootstrap=bootstrap),
            refute=False,
            execution=execution,
        )
    )


# %% [markdown]
# ## A plain execution is the baseline
#
# One `Execution` carries the seed, the thread budget, the memory limit, the observers and the
# cancellation token. The default one is enough to recover the effect.

# %%
baseline = run(ant.Execution(seed=3, threads=1))
print(f"baseline effect {baseline.answer.value:.3f} (truth {TRUE_EFFECT})")
print("calibration:", baseline.calibration.status)
assert abs(baseline.answer.value - TRUE_EFFECT) < TOLERANCE


# %% [markdown]
# ## Observe a run without changing it
#
# `on_stage` receives the name and a payload dict of each finished stage; `on_progress`
# receives a fraction and a stage name. A roomy `ResourceLimits` admits the run. None of these
# changes the estimate.

# %%
stages: list[str] = []
progress: list[tuple[float, str]] = []
watched = run(
    ant.Execution(
        seed=3,
        threads=1,
        limits=ant.ResourceLimits(memory_bytes=64_000_000),
        on_stage=lambda name, payload: stages.append(name),
        on_progress=lambda fraction, stage: progress.append((fraction, stage)),
    )
)
print("stages seen:", sorted(set(stages)))
assert stages
assert abs(watched.answer.value - baseline.answer.value) < 1e-12


# %% [markdown]
# ## A limit refuses before work
#
# A zero-byte admitted pool cannot hold even the preparation. The refusal is a typed
# `CausalResourceError`, raised before anything is fitted; it is an operational failure, not
# a scientific refusal, so it is an exception rather than an `Outcome`.

# %%
tiny = ant.Execution(limits=ant.ResourceLimits(memory_bytes=0), cache=ant.CachePolicy.disabled())
try:
    ant.prepare(
        data,
        graph=GRAPH,
        query=query,
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=tiny,
    )
except CausalResourceError as error:
    print("resource refusal:", type(error).__name__, "|", error)
else:
    raise AssertionError("a zero-byte memory limit should be refused")


# %% [markdown]
# ## Cancel before the call
#
# A token cancelled before the call stops it at the first boundary. A batch call raises a typed
# `CausalCancelledError` and publishes no claim.

# %%
token = ant.CancellationToken()
token.cancel()
assert token.is_cancelled()
try:
    ant.estimation.analyze_many(
        data,
        graph=GRAPH,
        queries=[query],
        execution=ant.Execution(cancel=token),
    )
except CausalCancelledError as error:
    print("cancelled:", type(error).__name__, "|", error)
else:
    raise AssertionError("a cancelled token should stop the call")


# %% [markdown]
# ## Cancel from an observer
#
# The application's own progress callback can cancel the run it is watching. Cancelling at the
# first bootstrap progress report stops the bootstrap early. The point estimate was already
# complete, and the result records that it was cancelled and how many replicates finished.

# %%
live = ant.CancellationToken()
seen = {"bootstrap": False}


def on_progress(fraction: float, stage: str) -> None:
    if stage == "bootstrap" and not seen["bootstrap"]:
        seen["bootstrap"] = True
        live.cancel()


full = run(ant.Execution(seed=3, threads=1), bootstrap=BOOTSTRAP)
stopped = run(
    ant.Execution(seed=3, threads=1, cancel=live, on_progress=on_progress), bootstrap=BOOTSTRAP
)
done = stopped.performance.bootstrap_replicates_ok or 0
print(
    f"full run {full.performance.bootstrap_replicates_ok}/{BOOTSTRAP} replicates; "
    f"cancelled run {done}/{BOOTSTRAP}, cancelled={stopped.performance.cancelled}"
)
assert full.performance.bootstrap_replicates_ok == BOOTSTRAP
assert stopped.performance.cancelled
assert done < BOOTSTRAP
assert abs(stopped.answer.value - TRUE_EFFECT) < TOLERANCE


# %% [markdown]
# ## Which build is this?
#
# `build_flavor()` returns "release" or "development". A release build refuses a route whose
# supporting evidence is not shipped. A development build runs such a route instead and records
# what a release build would have refused, so use it to explore unsupported routes and never to
# produce results you publish; artifacts it seals are stamped as development and release
# consumers refuse them. This example runs the same on either build.

# %%
flavor = ant.build_flavor()
print("build flavor:", flavor)
assert isinstance(flavor, str)
