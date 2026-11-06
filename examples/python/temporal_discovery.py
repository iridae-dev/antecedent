#!/usr/bin/env python3
"""Find lagged and same-time causal links in a multivariate time series.

Question: in a four-variable series, which variables drive which, and at what lag? Which
links are same-time (contemporaneous) and which reach back in time?

Data. One series of 1,500 steps from a declared vector autoregression. ``x``, ``y`` and ``z``
each depend on their own previous value, ``x[t-1] -> y[t]`` (0.6) and ``y[t-1] -> z[t]``
(0.6); ``w[t]`` is driven by the same-time value ``x[t]`` (0.8). A latent series ``h``
(an AR(1) process, never given to the search) feeds ``x`` and ``z`` one step later with
weight 0.5. The true lagged links are therefore ``x->x, y->y, z->z, x->y, y->z`` at lag 1,
and the only same-time link is ``x - w``.

What is checked. PCMCI+ (contemporaneous and lagged, assuming no latent variables) and LPCMCI
(allowing latent variables) both recover every true lagged link and the same-time ``x - w``
adjacency, checked against the declared coefficients. The latent ``h`` makes extra
adjacencies possible; they are printed with their marks, not asserted away. A statement
that lagged-only PCMCI cannot honour (a same-time required link) is refused with a typed
reason.

What this does NOT claim. Recovery here is on one seeded simulation with linear partial
correlation tests; it is not a coverage guarantee, and same-time orientations are not
asserted. No interval is reported.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent import discovery as disc
from antecedent.errors import CausalUnsupportedError

N, BURN = 1500, 200
NAMES = ["x", "y", "z", "w"]
TRUE_LAGGED = {
    ("x", "x", 1),
    ("y", "y", 1),
    ("z", "z", 1),
    ("x", "y", 1),
    ("y", "z", 1),
}  # (cause, effect, lag)
TRUE_SAME_TIME = {frozenset(("x", "w"))}
EXEC = ant.Execution(seed=5)


def simulate(seed: int = 3):
    rng = np.random.default_rng(seed)
    n = N + BURN
    h, x, y, z, w = (np.zeros(n) for _ in range(5))
    for t in range(1, n):
        h[t] = 0.7 * h[t - 1] + rng.normal()
        x[t] = 0.5 * x[t - 1] + 0.5 * h[t - 1] + rng.normal()
        y[t] = 0.5 * y[t - 1] + 0.6 * x[t - 1] + rng.normal()
        z[t] = 0.4 * z[t - 1] + 0.6 * y[t - 1] + 0.5 * h[t - 1] + rng.normal()
        w[t] = 0.8 * x[t] + 0.5 * rng.normal()
    series = {"x": x, "y": y, "z": z, "w": w}  # h is latent: not part of the table
    return NAMES, [series[name][BURN:] for name in NAMES]


def lagged(result) -> set[tuple[str, str, int]]:
    """(cause, effect, lag) for every adjacency across different times, ordered by time."""
    found = set()
    for e in result.graph_edges:
        if e.source_lag == e.target_lag:
            continue
        if e.source_lag > e.target_lag:  # the earlier node is the one with the larger lag
            found.add((e.source, e.target, e.source_lag - e.target_lag))
        else:
            found.add((e.target, e.source, e.target_lag - e.source_lag))
    return found


def same_time(result) -> set[frozenset]:
    return {
        frozenset((e.source, e.target))
        for e in result.graph_edges
        if e.source_lag == e.target_lag == 0
    }


names, cols = simulate()
print("series:", {n: round(float(c.std()), 2) for n, c in zip(names, cols, strict=True)})

# %% [markdown]
# ## PCMCI+: lagged and same-time links
#
# PCMCI+ tests each lag-window pair conditional on the estimated parents of both ends, so the
# autocorrelation in `x` does not manufacture a link into `y`. It also adjoins same-time
# pairs. Read `graph_edges` for adjacencies; the lag of an edge is the difference of its two
# end lags.

# %%
plus = disc.PCMCIPlus(max_lag=1, alpha=0.01, fdr=False).run((names, cols), execution=EXEC)
found = lagged(plus)
print("PCMCI+ lagged links:", sorted(found))
print("PCMCI+ same-time adjacencies:", sorted(sorted(p) for p in same_time(plus)))
assert TRUE_LAGGED <= found, sorted(TRUE_LAGGED - found)
assert TRUE_SAME_TIME <= same_time(plus)
print("extra lagged links (not in the declared model):", sorted(found - TRUE_LAGGED))

# %% [markdown]
# ## LPCMCI: allowing for a latent driver
#
# `h` feeds `x` and `z` but is not in the table. LPCMCI searches for a PAG, so an edge can
# carry circle marks where the data cannot rule out a latent common cause. It recovers the
# same declared lagged links.

# %%
lpcmci = disc.LPCMCI(max_lag=1, alpha=0.01, fdr=False).run((names, cols), execution=EXEC)
found_l = lagged(lpcmci)
print("LPCMCI lagged links:", sorted(found_l))
assert TRUE_LAGGED <= found_l, sorted(TRUE_LAGGED - found_l)
assert TRUE_SAME_TIME <= same_time(lpcmci)
print("circle edges in the PAG:", lpcmci.pag_circle_edges)
for e in lpcmci.graph_edges:
    if (e.source, e.source_lag, e.target, e.target_lag) in {
        ("x", 1, "z", 0),
        ("z", 0, "x", 1),
    }:
        print("x[t-1] ~ z[t] (the latent path):", e.at_source, e.at_target)

# %% [markdown]
# ## Check against the declared coefficients
#
# Each recovered lagged link is a claim about a declared coefficient. For the two cross links
# the lag-1 regression slope of the effect on its cause, adjusting for the effect's own past,
# should be near the declared 0.6. This is an independent least-squares reference, not part of
# the discovery.

# %%
col = dict(zip(names, cols, strict=True))


def lag1_slope(cause: str, effect: str) -> float:
    design = np.column_stack([col[cause][:-1], col[effect][:-1], np.ones(N - 1)])
    return float(np.linalg.lstsq(design, col[effect][1:], rcond=None)[0][0])


slopes = {("x", "y"): lag1_slope("x", "y"), ("y", "z"): lag1_slope("y", "z")}
print("lag-1 slopes:", {k: round(v, 3) for k, v in slopes.items()})
assert abs(slopes[("x", "y")] - 0.6) < 0.08
assert abs(slopes[("y", "z")] - 0.6) < 0.1  # the latent h adds a little bias to this one

# %% [markdown]
# ## A statement lagged-only PCMCI cannot honour
#
# PCMCI searches lagged links into the present. A required same-time link has no meaning for
# it, so it is refused with a reason code rather than ignored.

# %%
knowledge = disc.BackgroundKnowledge(required=(disc.Link("x", "w"),))
try:
    disc.PCMCI(max_lag=1, fdr=False, knowledge=knowledge).run((names, cols), execution=EXEC)
except CausalUnsupportedError as error:
    assert error.reason_code == "option_not_applicable", error
    print("refused:", error.reason_code or type(error).__name__, "-", error)
else:
    raise AssertionError("a same-time requirement was accepted by lagged-only PCMCI")
